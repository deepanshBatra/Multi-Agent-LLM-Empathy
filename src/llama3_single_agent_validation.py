"""
llama3_single_agent_validation.py
-----------------------------------
Single-agent per-utterance inference using the new dedicated endpoint.
Replicates the spirit of llama3_plain_validation.ipynb adapted for
the /predict API (which has a hard output cap — scene batching impossible).

Run: .venv\Scripts\python.exe -m src.llama3_single_agent_validation
"""

import os
import re
import json
import time
import random
import threading
import concurrent.futures

import pandas as pd
import requests
import google.auth
import google.auth.transport.requests
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV    = os.path.join(BASE_DIR, "data",  "test_sent_emo.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "logs",  "llama3_single_agent_results.csv")

PROJECT_ID    = "project-d92ffbcd-75f0-4c50-bca"
LOCATION      = "us-east1"
ENDPOINT_ID   = "mg-endpoint-a66f56b4-7b58-4560-b43e-2a8777c38cd9"
DEDICATED_URL = f"https://{ENDPOINT_ID}.{LOCATION}-574984131117.prediction.vertexai.goog"
PREDICT_URL   = (
    f"{DEDICATED_URL}/v1/projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/endpoints/{ENDPOINT_ID}:predict"
)

VALID_EMOTIONS = {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}
MAX_WORKERS  = 6     # parallel utterance calls
MAX_RETRIES  = 3
BASE_BACKOFF = 2.0
_sem         = threading.Semaphore(6)
_csv_lock    = threading.Lock()

# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
_creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
_auth_req = google.auth.transport.requests.Request()
_creds.refresh(_auth_req)
print(f"[Auth] Token obtained (length={len(_creds.token)})")


def _token() -> str:
    if not _creds.valid:
        _creds.refresh(_auth_req)
    return _creds.token


# ─────────────────────────────────────────────
# SAME SYSTEM PROMPT AS llama3_plain_validation
# (adapted: per-utterance instead of scene-batch)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert Emotion Recognition assistant specialized in the MELD benchmark "
    "(Friends TV show). Predict ONE emotion for the TARGET utterance. "
    "Labels: anger, disgust, fear, joy, neutral, sadness, surprise. "
    "Output ONLY: {\"predicted_emotion\": \"label\"}"
)


def build_prompt(context: str, speaker: str, utterance: str) -> str:
    """Builds the exact Llama 3 chat-template prompt used in the notebook."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Context:\n{context}\n\n"
        f"TARGET — {speaker}: \"{utterance}\"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def _call_raw(prompt: str) -> str:
    resp = requests.post(
        PREDICT_URL,
        json={"instances": [{
            "prompt": prompt,
            "max_tokens": 128,
            "temperature": 0.0
        }]},
        headers={"Authorization": f"Bearer {_token()}",
                 "Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["predictions"][0]
    # Strip prompt echo
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    if marker in raw:
        raw = raw.split(marker)[-1]
    raw = raw.strip()
    for prefix in ("Output:\n", "Output: "):
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break
    return raw


def _call_with_retry(prompt: str) -> str:
    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with _sem:
                return _call_raw(prompt)
        except Exception as e:
            last = e
            if attempt < MAX_RETRIES:
                wait = BASE_BACKOFF * (2 ** attempt) + random.uniform(0.1, 1.0)
                time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last}")


def parse_prediction(raw: str) -> str:
    """Extract the emotion label from the model's response."""
    raw = raw.strip()

    # Try JSON first
    try:
        m = re.search(r"\{[^}]+\}", raw)
        if m:
            obj = json.loads(m.group(0))
            label = (
                obj.get("predicted_emotion")
                or obj.get("vote")
                or obj.get("emotion")
                or ""
            )
            label = str(label).strip().lower()
            if label in VALID_EMOTIONS:
                return label
    except Exception:
        pass

    # Regex fallback — bare label word
    m = re.search(
        r"\b(neutral|joy|surprise|anger|sadness|disgust|fear)\b",
        raw, re.IGNORECASE
    )
    if m:
        return m.group(1).lower()

    return "neutral"  # last resort


# ─────────────────────────────────────────────
# DATA LOADING & SCENE GROUPING
# ─────────────────────────────────────────────
def load_and_group(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise key column names
    col_map = {c.lower(): c for c in df.columns}
    df = df.rename(columns={
        col_map.get("dialogue_id", "Dialogue_ID"):     "Dialogue_ID",
        col_map.get("utterance_id", "Utterance_ID"):   "Utterance_ID",
        col_map.get("speaker", "Speaker"):             "Speaker",
        col_map.get("utterance", "Utterance"):         "Utterance",
        col_map.get("emotion", "Emotion"):             "Emotion",
    })
    df["Recognition_ID"] = df["Dialogue_ID"].astype(str) + "_" + df["Utterance_ID"].astype(str)

    scenes = {}
    for _, row in df.iterrows():
        did = int(row["Dialogue_ID"])
        scenes.setdefault(did, []).append(row.to_dict())

    # Sort utterances within each scene
    for did in scenes:
        scenes[did].sort(key=lambda r: int(r.get("Utterance_ID", 0)))

    print(f"[Data] {len(df)} utterances across {len(scenes)} scenes")
    return df, scenes


def short_context(utterances: list, idx: int, n: int = 3) -> str:
    window = utterances[max(0, idx - n): idx]
    if not window:
        return "[Start of scene]"
    return "\n".join(f"{u['Speaker']}: {u['Utterance']}" for u in window)


# ─────────────────────────────────────────────
# RESUME SUPPORT
# ─────────────────────────────────────────────
def load_done_ids(output_file: str) -> set:
    if not os.path.exists(output_file):
        return set()
    df = pd.read_csv(output_file)
    return set(df["Recognition_ID"].astype(str).tolist())


# ─────────────────────────────────────────────
# SCENE RUNNER
# ─────────────────────────────────────────────
def run_scene(scene_utts: list, done_ids: set) -> list:
    results = []
    for idx, u in enumerate(scene_utts):
        rec_id = str(u.get("Recognition_ID", ""))
        if rec_id in done_ids:
            continue

        utterance  = str(u.get("Utterance", ""))
        speaker    = str(u.get("Speaker", "Unknown"))
        actual_emo = str(u.get("Emotion", "unknown")).strip().lower()
        context    = short_context(scene_utts, idx)

        prompt = build_prompt(context, speaker, utterance)

        try:
            raw = _call_with_retry(prompt)
            predicted = parse_prediction(raw)
        except Exception as e:
            print(f"  [Error] {rec_id}: {e}")
            predicted = "neutral"

        results.append({
            "Dialogue_ID":    u.get("Dialogue_ID"),
            "Utterance_ID":   u.get("Utterance_ID"),
            "Recognition_ID": rec_id,
            "Speaker":        speaker,
            "Utterance":      utterance,
            "Actual_Emotion": actual_emo,
            "Predicted_Emotion": predicted,
        })

    return results


def save_rows(rows: list, output_file: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    file_exists = os.path.exists(output_file)
    with _csv_lock:
        pd.DataFrame(rows).to_csv(
            output_file, mode="a", index=False,
            header=not file_exists
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Llama 3 Single-Agent Validation — MELD Test Set")
    print(f"  Output → {OUTPUT_FILE}")
    print("=" * 65)

    df, scenes = load_and_group(TEST_CSV)
    done_ids   = load_done_ids(OUTPUT_FILE)

    scene_ids = sorted(scenes.keys())
    todo = [sid for sid in scene_ids
            if any(str(u.get("Recognition_ID","")) not in done_ids
                   for u in scenes[sid])]

    print(f"[Info] {len(todo)}/{len(scene_ids)} scenes to process.")

    def _run(sid):
        rows = run_scene(scenes[sid], done_ids)
        save_rows(rows, OUTPUT_FILE)
        return len(rows)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_run, sid): sid for sid in todo}
        for f in tqdm(concurrent.futures.as_completed(futures),
                      total=len(todo), unit="scene"):
            futures.pop(f)

    print("\n[Done] Inference complete.")
    print(f"       Results: {OUTPUT_FILE}")

    # ── Evaluation ─────────────────────────────────────────────
    results_df = pd.read_csv(OUTPUT_FILE)
    results_df = results_df.drop_duplicates(subset=["Recognition_ID"], keep="last")

    y_true = results_df["Actual_Emotion"].str.strip().str.lower()
    y_pred = results_df["Predicted_Emotion"].str.strip().str.lower()

    # Keep only valid labels
    mask = y_true.isin(VALID_EMOTIONS) & y_pred.isin(VALID_EMOTIONS)
    y_true, y_pred = y_true[mask], y_pred[mask]

    wf1 = f1_score(y_true, y_pred, average="weighted", labels=sorted(VALID_EMOTIONS))

    print("\n" + "=" * 65)
    print(f"  Weighted F1:  {wf1:.4f}  (baseline: 0.7771)")
    print("=" * 65)
    print(classification_report(
        y_true, y_pred,
        labels=sorted(VALID_EMOTIONS),
        zero_division=0
    ))

    print("\nPrediction distribution:")
    print(results_df["Predicted_Emotion"].value_counts().to_string())


if __name__ == "__main__":
    main()
