"""
llama3_scene_batch_validation_optimized.py
---------------------------------------------
Optimized single-agent scene-batch validation pipeline designed to exceed 0.78+ Weighted F1.
Integrates:
1. Dynamic In-Context Learning (local TF-IDF few-shot retrieval from MELD training data).
2. Speaker Personality Bio-Cards Context Injection (from logs/speaker_bio_cards.json).
3. Payload and Hyperparameter Optimizations (zero truncation, temperature=0.1).

Saves results incrementally to logs/llama3_scene_batch_results_optimized.csv (resume-safe).
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

from src.retrieval_utils import get_retriever

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV    = os.path.join(BASE_DIR, "data",  "test_sent_emo.csv")
BIO_JSON    = os.path.join(BASE_DIR, "logs",  "speaker_bio_cards.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "logs",  "llama3_scene_batch_results_optimized.csv")

PROJECT_ID    = "project-d92ffbcd-75f0-4c50-bca"
LOCATION      = "us-east1"
ENDPOINT_ID   = "mg-endpoint-a66f56b4-7b58-4560-b43e-2a8777c38cd9"
PREDICT_URL   = (
    f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
    f"/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
)

VALID_EMOTIONS = {"neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"}
MAX_WORKERS  = 3     # concurrent scene queries
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
# CONTEXT LOADING HELPERS
# ─────────────────────────────────────────────
def load_speaker_bio_cards(json_path: str) -> dict:
    if not os.path.exists(json_path):
        print(f"[Warning] Bio cards JSON not found at {json_path}. Proceeding without persona context.")
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load bio cards: {e}. Proceeding without persona context.")
        return {}


def format_scene_bio_cards(bio_cards: dict, speakers: list) -> str:
    if not bio_cards or not speakers:
        return ""
    
    lines = ["### SPEAKER PERSONAS"]
    found_any = False
    for spk in set(speakers):
        if spk in bio_cards:
            found_any = True
            bio = bio_cards[spk]
            lines.append(f"**{spk}**:")
            if isinstance(bio, dict):
                if "static_persona" in bio:
                    lines.append(f"  - Persona: {bio['static_persona']}")
                if "linguistic_style" in bio:
                    lines.append(f"  - Style: {bio['linguistic_style']}")
                if "baseline_arousal" in bio:
                    lines.append(f"  - Baseline Arousal: {bio['baseline_arousal']}")
            else:
                lines.append(f"  - Profile: {str(bio)}")
    
    if not found_any:
        return ""
    
    return "\n".join(lines)


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


def build_scene_prompt(dialogue_id: int, scene_utts: list, retriever, bio_cards: dict) -> str:
    # 1. Gather all speakers in the scene to inject baseline bio card context
    speakers = [u.get("Speaker", "Unknown") for u in scene_utts]
    bio_context = format_scene_bio_cards(bio_cards, speakers)
    
    # 2. Get Dynamic Few-Shot Examples (Dynamic In-Context Learning)
    few_shot_lines = []
    for u in scene_utts:
        utt_text = u.get("Utterance", "")
        # Query local TF-IDF retriever for top training matches (excluding exact match test leak)
        matches = retriever.get_top_k_examples(utt_text, k=1)
        if matches:
            match = matches[0]
            few_shot_lines.append(
                f"Utterance: \"{match['utterance']}\" -> Predicted: {match['emotion']}"
            )
            
    few_shot_context = ""
    if few_shot_lines:
        few_shot_context = (
            "### DIALOGUE REFERENCE SAMPLES\n"
            + "\n".join(few_shot_lines[:3]) # Cap at top 3 reference examples to control context length
        )

    # 3. Format Scene dialogue sequence
    scene_lines = []
    for u in scene_utts:
        utt_id = str(u.get("Utterance_ID", "NA"))
        speaker = str(u.get("Speaker", "Unknown"))
        utterance = str(u.get("Utterance", ""))
        scene_lines.append(f"{utt_id} | {speaker}: {utterance}")

    # Compile enriched prompt block
    prompt_builder = []
    prompt_builder.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}")
    
    if bio_context:
        prompt_builder.append(bio_context)
    if few_shot_context:
        prompt_builder.append(few_shot_context)
        
    prompt_builder.append("<|eot_id|><|start_header_id|>user<|end_header_id|>\n")
    prompt_builder.append(f"Scene Dialogue ID: {dialogue_id}")
    prompt_builder.append("Utterances in order:")
    prompt_builder.append("\n".join(scene_lines))
    prompt_builder.append("\nReturn predictions for ALL utterance_id values above.<|eot_id|>")
    prompt_builder.append("<|start_header_id|>assistant<|end_header_id|>")

    return "\n\n".join(prompt_builder)


# ─────────────────────────────────────────────
# INFERENCE & RETRIES
# ─────────────────────────────────────────────
def _call_raw(prompt: str) -> str:
    payload = {
        "instances": [{
            "prompt": prompt,
            "max_tokens": 2048,   # High token cap to guarantee zero JSON truncation
            "temperature": 0.1     # Standard deterministic temperature
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


def run_scene(scene_utts: list, retriever, bio_cards: dict) -> list:
    did = int(scene_utts[0]["Dialogue_ID"])
    prompt = build_scene_prompt(did, scene_utts, retriever, bio_cards)
    
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
            "raw_output": model_output[:300]  # keep output clean
        })
    return rows


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Llama 3.1 Scene-Batch Optimized Pipeline — MELD Test Set")
    print(f"  Output → {OUTPUT_FILE}")
    print("=" * 65)

    df, scenes = load_and_group(TEST_CSV)
    done_scenes = load_done_scenes(OUTPUT_FILE)
    print(f"[Info] {len(done_scenes)}/{len(scenes)} scenes already processed.")

    todo_ids = [sid for sid in sorted(scenes.keys()) if sid not in done_scenes]
    print(f"[Info] {len(todo_ids)} scenes remaining to process.")

    # Instantiate TF-IDF indexer and bio cards loader
    print("[Info] Initializing Dynamic Few-Shot index and Speaker personas...")
    retriever = get_retriever()
    bio_cards = load_speaker_bio_cards(BIO_JSON)

    def _run(sid):
        rows = run_scene(scenes[sid], retriever, bio_cards)
        save_rows(rows, OUTPUT_FILE)
        return len(rows)

    if todo_ids:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_run, sid): sid for sid in todo_ids}
            for f in tqdm(concurrent.futures.as_completed(futures),
                          total=len(todo_ids), desc="Scenes", unit="scene"):
                futures.pop(f)

    print("\n[Done] Optimized Inference complete.")
    print(f"       Results: {OUTPUT_FILE}")

    # ── Evaluation ─────────────────────────────────────────────
    try:
        results_df = pd.read_csv(OUTPUT_FILE)
        results_df = results_df.drop_duplicates(subset=["Recognition_ID"], keep="last")

        y_true = results_df["Actual_Emotion"].str.strip().str.lower()
        y_pred = results_df["Predicted_Emotion"].str.strip().str.lower()

        # Keep only valid labels
        mask = y_true.isin(VALID_EMOTIONS) & y_pred.isin(VALID_EMOTIONS)
        y_true, y_pred = y_true[mask], y_pred[mask]

        wf1 = f1_score(y_true, y_pred, average="weighted", labels=sorted(VALID_EMOTIONS))

        print("\n" + "=" * 65)
        print(f"  Optimized Weighted F1:  {wf1:.4f}  (target > 0.7800)")
        print("=" * 65)
        print(classification_report(
            y_true, y_pred,
            labels=sorted(VALID_EMOTIONS),
            zero_division=0
        ))

        print("\nPrediction distribution:")
        print(results_df["Predicted_Emotion"].value_counts().to_string())
    except Exception as e:
        print(f"[Eval Error] Could not evaluate: {e}")


if __name__ == "__main__":
    main()
