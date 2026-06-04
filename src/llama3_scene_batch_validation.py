"""
llama3_scene_batch_validation.py
-----------------------------------
Single-agent scene-batch inference using the dedicated Vertex AI endpoint.
Replicates the exact prompt and logic of llama3_plain_validation.ipynb
adapted to use the correct vLLM instances payload format to avoid truncation.

Run: .venv\\Scripts\\python.exe -m src.llama3_scene_batch_validation
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
OUTPUT_FILE = os.path.join(BASE_DIR, "logs",  "llama3_scene_batch_results.csv")

PROJECT_ID    = "project-d92ffbcd-75f0-4c50-bca"
LOCATION      = "us-east1"
ENDPOINT_ID   = "mg-endpoint-a66f56b4-7b58-4560-b43e-2a8777c38cd9"
DEDICATED_URL = f"https://{ENDPOINT_ID}.{LOCATION}-574984131117.prediction.vertexai.goog"
PREDICT_URL   = (
    f"{DEDICATED_URL}/v1/projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/endpoints/{ENDPOINT_ID}:predict"
)

VALID_EMOTIONS = {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}
MAX_WORKERS  = 3     # parallel scene queries
MAX_RETRIES  = 4
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
# SCENE PROMPT BUILDING
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert Emotion Recognition assistant specialized in the MELD benchmark.\n"
    "For each scene, predict one emotion for EVERY utterance using labels: [anger, disgust, fear, joy, neutral, sadness, surprise].\n"
    "Return JSON only in this schema:\n"
    "{\n"
    "  \"predictions\": [\n"
    "    {\"utterance_id\": \"id\", \"predicted_emotion\": \"label\", \"reasoning\": \"short reason\"}\n"
    "  ]\n"
    "}"
)


def build_scene_prompt(dialogue_id: int, scene_utts: list) -> str:
    scene_lines = []
    for u in scene_utts:
        utt_id = str(u.get("Utterance_ID", "NA"))
        speaker = str(u.get("Speaker", "Unknown"))
        utterance = str(u.get("Utterance", ""))
        scene_lines.append(f"{utt_id} | {speaker}: {utterance}")

    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Scene Dialogue ID: {dialogue_id}\n"
        f"Utterances in order:\n"
        + "\n".join(scene_lines)
        + f"\n\nReturn predictions for ALL utterance_id values above.<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )


# ─────────────────────────────────────────────
# INFERENCE & RETRIES
# ─────────────────────────────────────────────
def _call_raw(prompt: str) -> str:
    # Key fix: Pass max_tokens inside the instances dict so the endpoint respects it!
    payload = {
        "instances": [{
            "prompt": prompt,
            "max_tokens": 1024,
            "temperature": 0.1
        }]
    }
    resp = requests.post(
        PREDICT_URL,
        json=payload,
        headers={"Authorization": f"Bearer {_token()}",
                 "Content-Type": "application/json"},
        timeout=180,
    )
    resp.raise_for_status()
    raw = resp.json()["predictions"][0]
    # Strip prompt echo
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    if marker in raw:
        raw = raw.split(marker)[-1]
    raw = raw.strip()
    for prefix in ("Output:\n", "Output: ", "output:\n", "output: "):
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


# ─────────────────────────────────────────────
# PARSING SCENE PREDICTIONS
# ─────────────────────────────────────────────
def parse_scene_predictions(text: str) -> dict:
    """Return mapping: utterance_id(str) -> {predicted_emotion, reasoning}."""
    if not text:
        return {}

    parsed = None
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            parsed = json.loads(match.group(0))
    except Exception:
        parsed = None

    if parsed is None:
        return {}

    predictions = []
    if isinstance(parsed, dict):
        maybe_list = parsed.get("predictions", [])
        if isinstance(maybe_list, list):
            predictions = maybe_list
    elif isinstance(parsed, list):
        predictions = parsed

    out = {}
    for item in predictions:
        if not isinstance(item, dict):
            continue
        utt_id = item.get("utterance_id")
        if utt_id is None:
            continue
        emotion = item.get("predicted_emotion") or item.get("emotion")
        reason = item.get("reasoning", "")
        out[str(utt_id)] = {
            "predicted_emotion": emotion.lower().strip() if isinstance(emotion, str) else "neutral",
            "reasoning": str(reason) if reason is not None else ""
        }
    return out


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


def load_done_scenes(output_file: str) -> set:
    if not os.path.exists(output_file):
        return set()
    try:
        df = pd.read_csv(output_file)
        if "Dialogue_ID" in df.columns:
            return set(df["Dialogue_ID"].dropna().unique().astype(int))
    except Exception:
        pass
    return set()


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


def run_scene(scene_utts: list) -> list:
    did = int(scene_utts[0]["Dialogue_ID"])
    prompt = build_scene_prompt(did, scene_utts)
    
    try:
        model_output = _call_with_retry(prompt)
        pred_map = parse_scene_predictions(model_output)
    except Exception as exc:
        print(f"\n[Error] Scene {did} failed: {exc}")
        pred_map = {}
        model_output = f"ERROR: {exc}"

    rows = []
    for u in scene_utts:
        utt_id = str(u["Utterance_ID"])
        pred_item = pred_map.get(utt_id, {})
        predicted_emo = pred_item.get("predicted_emotion", "neutral")
        reasoning = pred_item.get("reasoning", "")
        
        rows.append({
            "Dialogue_ID": u["Dialogue_ID"],
            "Utterance_ID": u["Utterance_ID"],
            "Recognition_ID": u["Recognition_ID"],
            "Speaker": u["Speaker"],
            "Utterance": u["Utterance"],
            "Actual_Emotion": u["Emotion"],
            "Predicted_Emotion": predicted_emo,
            "Reasoning": reasoning,
            "raw_output": model_output[:300]  # truncate to keep CSV clean
        })
    return rows


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Llama 3.1 Scene-Batch Validation — MELD Test Set")
    print(f"  Output → {OUTPUT_FILE}")
    print("=" * 65)

    df, scenes = load_and_group(TEST_CSV)
    done_scenes = load_done_scenes(OUTPUT_FILE)
    print(f"[Info] {len(done_scenes)}/{len(scenes)} scenes already processed.")

    todo_ids = [sid for sid in sorted(scenes.keys()) if sid not in done_scenes]
    print(f"[Info] {len(todo_ids)} scenes remaining to process.")

    def _run(sid):
        rows = run_scene(scenes[sid])
        save_rows(rows, OUTPUT_FILE)
        return len(rows)

    if todo_ids:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_run, sid): sid for sid in todo_ids}
            for f in tqdm(concurrent.futures.as_completed(futures),
                          total=len(todo_ids), desc="Scenes", unit="scene"):
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
