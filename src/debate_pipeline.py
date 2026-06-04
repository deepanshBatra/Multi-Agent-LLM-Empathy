"""
debate_pipeline.py
------------------
3-Round Multi-Agent Debate Pipeline for MELD Emotion Recognition.

Architecture (per scene):
  ROUND 1  [Parallel] — 3 independent agents vote on ALL utterances in the scene
    ├── Agent 1: Plain Scene Classifier   (scene narrative lens)
    ├── Agent 2: Linguistic Analyst       (word/tone/punctuation lens)
    └── Agent 3: Temporal/Shift Analyst   (emotional momentum lens)

  Per utterance after Round 1:
    - Unanimous (3/3 agree)   → accept immediately, 0 extra calls
    - Majority  (2/3 agree)   → accept majority,    0 extra calls
    - 3-way split             → ROUND 2: Arbitrator reads all 3 votes + reasons

  Saves incrementally to logs/debate_pipeline_results.csv (resume-safe).
"""

import os
import re
import json
import time
import random
import threading
import concurrent.futures
from collections import Counter

import pandas as pd
from tqdm import tqdm

import requests
import google.auth
import google.auth.transport.requests
from dotenv import load_dotenv

from src.load_data import load_data_from_csv
from src.retrieval_utils import get_retriever

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MAX_SCENE_WORKERS    = 3   # scenes processed in parallel
MAX_CONCURRENT_CALLS = 6   # hard cap on simultaneous LLM requests
MAX_RETRIES          = 4
BASE_BACKOFF_S       = 1.5

_RETRYABLE_TOKENS = (
    "429", "503", "500", "quota", "resource",
    "timeout", "unavailable", "deadline", "overload", "rate",
)

_api_semaphore = threading.Semaphore(MAX_CONCURRENT_CALLS)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data",  "test_sent_emo.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "logs",  "debate_pipeline_results.csv")
PROMPTS_DIR = os.path.join(BASE_DIR, "src",   "llama3_prompts")

MELD_EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]

# ─────────────────────────────────────────────
# VERTEX AI / LLAMA 3.1 SETUP
# ─────────────────────────────────────────────
# Dedicated endpoint config (us-east1, fine-tuned Llama 3 ERC model)
PROJECT_ID    = "project-d92ffbcd-75f0-4c50-bca"
LOCATION      = "us-east1"
ENDPOINT_ID   = "mg-endpoint-a66f56b4-7b58-4560-b43e-2a8777c38cd9"
DEDICATED_URL = f"https://{ENDPOINT_ID}.{LOCATION}-574984131117.prediction.vertexai.goog"
PREDICT_URL   = (
    f"{DEDICATED_URL}/v1/projects/{PROJECT_ID}/locations/{LOCATION}"
    f"/endpoints/{ENDPOINT_ID}:predict"
)

print(f"[Config] Project: {PROJECT_ID} | Location: {LOCATION}")
print(f"[Config] Endpoint: {ENDPOINT_ID}")
print(f"[Config] Predict URL: {PREDICT_URL}")

# ── Google Auth ADC (auto-refreshes on expiry) ─────────────────
_gcp_creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
_gcp_auth_req = google.auth.transport.requests.Request()
_gcp_creds.refresh(_gcp_auth_req)
print(f"[Config] ADC token obtained (length={len(_gcp_creds.token)})")

# ─────────────────────────────────────────────
# PROMPT LOADING
# ─────────────────────────────────────────────
def _load(fname: str) -> str:
    with open(os.path.join(PROMPTS_DIR, fname), "r", encoding="utf-8") as f:
        return f.read()

PROMPTS = {
    "vote_plain":         _load("vote_plain.txt"),
    "vote_linguistic":    _load("vote_linguistic.txt"),
    "vote_context_shift": _load("vote_context_shift.txt"),
    "arbitrator":         _load("arbitrator.txt"),
}

# ─────────────────────────────────────────────
# LLM CALLER + RETRY
# ─────────────────────────────────────────────
def _get_token() -> str:
    """Returns a valid ADC access token, refreshing if expired."""
    if not _gcp_creds.valid:
        _gcp_creds.refresh(_gcp_auth_req)
    return _gcp_creds.token


def _call_llama_raw(full_prompt: str, max_tokens: int = 2048) -> str:
    """
    POSTs the pre-formatted Llama 3 chat-template prompt to the dedicated
    endpoint's /predict route and returns the generated text.
    max_tokens is dynamic: scale with scene size to prevent mid-JSON truncation.
    """
    payload = {
        "instances": [{
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1
        }],
    }
    resp = requests.post(
        PREDICT_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {_get_token()}",
            "Content-Type":  "application/json",
        },
        timeout=180,
    )
    resp.raise_for_status()
    body = resp.json()
    raw = body["predictions"][0]
    # Strip echoed prompt: take everything after the last assistant header
    split_marker = "<|start_header_id|>assistant<|end_header_id|>"
    if split_marker in raw:
        raw = raw.split(split_marker)[-1]
    raw = raw.strip()
    # Strip the "Output:\n" or "Output: " prefix the model sometimes prepends
    for prefix in ("Output:\n", "Output: ", "output:\n", "output: "):
        if raw.startswith(prefix):
            raw = raw[len(prefix):].strip()
            break
    return raw


def _call_with_retry(full_prompt: str, max_tokens: int = 2048) -> str:
    """Wraps _call_llama_raw with semaphore + exponential backoff."""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with _api_semaphore:
                return _call_llama_raw(full_prompt, max_tokens=max_tokens)
        except Exception as exc:
            last_exc = exc
            err_lower = str(exc).lower()
            is_retryable = any(tok in err_lower for tok in _RETRYABLE_TOKENS)
            if attempt < MAX_RETRIES and is_retryable:
                wait = BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0.1, 1.0)
                print(f"    [Retry {attempt+1}/{MAX_RETRIES}] {str(exc)[:80]} -- sleeping {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {last_exc}")

# ─────────────────────────────────────────────
# JSON / FIELD PARSERS
# ─────────────────────────────────────────────
def _extract_json(raw: str) -> dict | list:
    """Multi-strategy JSON extraction. Returns dict or list, never raises."""
    raw = str(raw).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw).strip()

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Last {...} block
    for m in reversed(list(re.finditer(r"\{[\s\S]*?\}", raw))):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue

    # Greedy nested-brace
    start = raw.find("{")
    if start != -1:
        depth, end = 0, -1
        for i, ch in enumerate(raw[start:], start):
            depth += (ch == "{") - (ch == "}")
            if depth == 0:
                end = i
                break
        if end != -1:
            try:
                return json.loads(raw[start: end + 1])
            except json.JSONDecodeError:
                pass

    return {}


def _canon_emotion(s: str) -> str:
    """Canonicalise to a MELD label or 'neutral'."""
    s = str(s).lower().strip()
    return s if s in MELD_EMOTIONS else "neutral"


def _parse_vote_response(raw: str) -> dict:
    """
    Parse a scene-level vote response.
    Returns {utterance_id: {"vote": str, "confidence": float, "reason": str}}.
    """
    parsed = _extract_json(raw)

    # Handle {"predictions": [...]} or bare [...]
    if isinstance(parsed, dict):
        items = parsed.get("predictions", [])
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    result = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        uid = str(item.get("utterance_id", item.get("id", ""))).strip()
        if not uid:
            continue
        vote = _canon_emotion(item.get("vote") or item.get("predicted_emotion") or "")
        conf_raw = item.get("confidence", 0.5)
        try:
            conf = float(max(0.0, min(1.0, conf_raw)))
        except (TypeError, ValueError):
            conf = 0.5
        reason = str(item.get("reason") or item.get("reasoning") or "").strip()
        result[uid] = {"vote": vote, "confidence": conf, "reason": reason}
    return result


def _parse_arbitrator_response(raw: str) -> dict:
    """Parse the Arbitrator's final verdict."""
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        parsed = {}

    emotion = _canon_emotion(
        parsed.get("predicted_emotion") or parsed.get("vote") or ""
    )
    # Fallback: scan raw text
    if emotion == "neutral" and "predicted_emotion" not in str(parsed):
        m = re.search(
            r'"?predicted_emotion"?\s*[:\-]\s*"?([a-z]+)"?', raw, re.IGNORECASE
        )
        if m:
            emotion = _canon_emotion(m.group(1))

    conf_raw = parsed.get("confidence", 0.5)
    try:
        conf = float(max(0.0, min(1.0, conf_raw)))
    except (TypeError, ValueError):
        conf = 0.5

    return {
        "predicted_emotion": emotion,
        "confidence": conf,
        "winning_agent":  parsed.get("winning_agent", "unknown"),
        "reasoning":      parsed.get("reasoning", ""),
    }

# ─────────────────────────────────────────────
# SCENE BUILDER HELPERS
# ─────────────────────────────────────────────
def _build_scene_text(utterances: list) -> str:
    """
    Formats scene utterances for injection into vote prompts.
    Each line: [utterance_id] Speaker: "text"
    """
    lines = []
    for u in utterances:
        uid = u.get("Recognition_ID", "?")
        spk = u.get("Speaker", "Unknown")
        txt = u.get("Utterance", "")
        lines.append(f'[{uid}] {spk}: "{txt}"')
    return "\n".join(lines)


def _short_history(utterances: list, idx: int, n: int = 2) -> str:
    """Last n turns before idx, formatted as 'Speaker: text'."""
    window = utterances[max(0, idx - n): idx]
    if not window:
        return "[Start of scene]"
    return "\n".join(f"{u['Speaker']}: {u['Utterance']}" for u in window)

# ─────────────────────────────────────────────
# ROUND 1 — AGENT CALLERS (scene-level)
# ─────────────────────────────────────────────
def _call_vote_agent_utterance(
    prompt_key: str,
    context: str,
    speaker: str,
    utterance: str,
    retrieved_examples: str = "",
) -> dict:
    """
    Per-utterance vote call — outputs a single compact JSON object.
    The endpoint has a hard deployment-time output cap (~50 chars), so
    scene-batching JSON arrays are impossible. Per-utterance calls work.
    """
    full_prompt = (
        PROMPTS[prompt_key]
        .replace("{CONTEXT}",            context)
        .replace("{SPEAKER}",            speaker)
        .replace("{UTTERANCE}",          utterance)
        .replace("{RETRIEVED_EXAMPLES}", retrieved_examples or "[No examples available]")
    )
    raw = _call_with_retry(full_prompt)
    return _parse_single_vote(raw)


def _parse_single_vote(raw: str) -> dict:
    """
    Parse a compact single-utterance vote response.
    Returns {"vote": str, "confidence": float, "reason": str}.
    """
    raw = str(raw).strip()

    # Try direct JSON parse first
    parsed = _extract_json(raw)

    if isinstance(parsed, dict):
        # Could be {"vote":...} directly OR {"predictions":[{...}]} fallback
        vote_raw = (
            parsed.get("vote")
            or parsed.get("predicted_emotion")
            or parsed.get("emotion")
            or ""
        )
        # Handle nested predictions array
        if not vote_raw and "predictions" in parsed:
            first = parsed["predictions"]
            if isinstance(first, list) and first:
                vote_raw = first[0].get("vote", "")
                parsed = first[0]

        vote = _canon_emotion(vote_raw)
        conf_raw = parsed.get("confidence", 0.5)
        try:
            conf = float(max(0.0, min(1.0, conf_raw)))
        except (TypeError, ValueError):
            conf = 0.5
        reason = str(parsed.get("reason") or parsed.get("reasoning") or "").strip()
        return {"vote": vote, "confidence": conf, "reason": reason}

    # Regex fallback — extract bare label
    m = re.search(
        r'"?(?:vote|predicted_emotion|emotion)"?\s*[:\-]\s*"?([a-z]+)"?',
        raw, re.IGNORECASE
    )
    vote = _canon_emotion(m.group(1)) if m else "neutral"
    return {"vote": vote, "confidence": 0.5, "reason": "[regex fallback]"}


# ─────────────────────────────────────────────
# ROUND 2 — ARBITRATOR (per utterance, on split)
# ─────────────────────────────────────────────
def _call_arbitrator(
    rec_id: str,
    utterance: str,
    speaker: str,
    short_hist: str,
    vote_plain: dict,
    vote_ling: dict,
    vote_ctx: dict,
) -> dict:
    """
    Call the Arbitrator when agents split 3-way.
    Uses compact prompt — outputs {\"vote\":...,\"confidence\":...} only.
    """
    full_prompt = (
        PROMPTS["arbitrator"]
        .replace("{UTTERANCE}",       str(utterance))
        .replace("{SPEAKER}",         str(speaker))
        .replace("{CONTEXT}",         str(short_hist))
        .replace("{VOTE_PLAIN}",      str(vote_plain.get("vote", "neutral")))
        .replace("{CONF_PLAIN}",      f"{vote_plain.get('confidence', 0.5):.2f}")
        .replace("{VOTE_LINGUISTIC}", str(vote_ling.get("vote", "neutral")))
        .replace("{CONF_LINGUISTIC}", f"{vote_ling.get('confidence', 0.5):.2f}")
        .replace("{VOTE_CONTEXT}",    str(vote_ctx.get("vote", "neutral")))
        .replace("{CONF_CONTEXT}",    f"{vote_ctx.get('confidence', 0.5):.2f}")
    )
    raw = _call_with_retry(full_prompt)
    return _parse_arbitrator_response(raw)

# ─────────────────────────────────────────────
# VOTE AGGREGATION
# ─────────────────────────────────────────────
def _aggregate_votes(
    votes: dict[str, dict]  # {"plain": {...}, "linguistic": {...}, "context": {...}}
) -> tuple[str, float, str, bool]:
    """
    Returns (predicted_emotion, avg_confidence, outcome_label, is_split).
    outcome_label: "unanimous" | "majority" | "split"
    """
    labels = [v["vote"] for v in votes.values() if v]
    counts = Counter(labels)
    most_common_label, most_common_count = counts.most_common(1)[0]

    if most_common_count == 3:
        outcome = "unanimous"
        confs = [v["confidence"] for v in votes.values() if v]
        return most_common_label, sum(confs) / len(confs), outcome, False

    if most_common_count == 2:
        outcome = "majority"
        majority_confs = [
            v["confidence"] for k, v in votes.items() if v and v["vote"] == most_common_label
        ]
        return most_common_label, sum(majority_confs) / len(majority_confs), outcome, False

    # 3-way split — no majority
    return most_common_label, 0.5, "split", True

# ─────────────────────────────────────────────
# SCENE-LEVEL PIPELINE
# ─────────────────────────────────────────────
def run_debate_scene(scene_obj: dict, retriever) -> list:
    utterances  = scene_obj["utterances"]
    dialogue_id = scene_obj["dialogue_id"]

    print(f"  [Scene {dialogue_id}] {len(utterances)} utterances — per-utterance debate...", flush=True)

    results = []

    for idx, u in enumerate(utterances):
        rec_id     = u.get("Recognition_ID", f"{dialogue_id}_{idx}")
        utterance  = u.get("Utterance", "")
        speaker    = u.get("Speaker", "Unknown")
        actual_emo = u.get("Emotion", "unknown")
        uid        = str(rec_id)

        # Context: last 3 turns before this utterance
        context = _short_history(utterances, idx, n=3)

        # Calibration examples for this specific utterance (train data only)
        examples     = retriever.get_top_k_examples(utterance, k=3)
        retrieved_ex = retriever.format_for_prompt(examples)

        # ── ROUND 1: 3 agents in parallel for THIS utterance ─────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            f_plain = pool.submit(
                _call_vote_agent_utterance,
                "vote_plain", context, speaker, utterance, retrieved_ex
            )
            f_ling = pool.submit(
                _call_vote_agent_utterance,
                "vote_linguistic", context, speaker, utterance, ""
            )
            f_ctx = pool.submit(
                _call_vote_agent_utterance,
                "vote_context_shift", context, speaker, utterance, ""
            )
            vp = f_plain.result()
            vl = f_ling.result()
            vc = f_ctx.result()

        # ── Aggregate ────────────────────────────────────────────────────────
        predicted_emo, avg_conf, outcome, is_split = _aggregate_votes(
            {"plain": vp, "linguistic": vl, "context": vc}
        )

        winning_agent = "majority"
        arbitrator_reasoning = ""

        # ── ROUND 2: Arbitrator (only on 3-way split) ────────────────────────
        if is_split:
            print(f"    [SPLIT] {uid} — plain={vp['vote']} ling={vl['vote']} ctx={vc['vote']} — calling Arbitrator", flush=True)
            short_hist = _short_history(utterances, idx)
            verdict = _call_arbitrator(
                rec_id=uid,
                utterance=utterance,
                speaker=speaker,
                short_hist=short_hist,
                vote_plain=vp,
                vote_ling=vl,
                vote_ctx=vc,
            )
            predicted_emo        = verdict["predicted_emotion"]
            avg_conf             = verdict["confidence"]
            winning_agent        = verdict["winning_agent"]
            arbitrator_reasoning = verdict["reasoning"]
            outcome              = "arbitrated"

        print(
            f"    [{uid}] {outcome:12s} | pred={predicted_emo:<9s} conf={avg_conf:.2f} | actual={actual_emo}",
            flush=True,
        )

        results.append({
            "Dialogue_ID":            dialogue_id,
            "Recognition_ID":         rec_id,
            "Speaker":                speaker,
            "Utterance":              utterance,
            "Actual_Emotion":         actual_emo,
            "predicted_emotion":      predicted_emo,
            "confidence":             round(avg_conf, 4),
            "outcome":                outcome,           # unanimous / majority / arbitrated
            "winning_agent":          winning_agent,
            "vote_plain":             vp["vote"],
            "vote_linguistic":        vl["vote"],
            "vote_context":           vc["vote"],
            "reason_plain":           vp["reason"][:200],
            "reason_linguistic":      vl["reason"][:200],
            "reason_context":         vc["reason"][:200],
            "arbitrator_reasoning":   arbitrator_reasoning[:300],
        })

    return results

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 65, flush=True)
    print("  Multi-Agent Debate Pipeline — MELD Test Set", flush=True)
    print(f"  Output → {OUTPUT_FILE}", flush=True)
    print(f"  Endpoint: {ENDPOINT_ID}", flush=True)
    print("=" * 65, flush=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load test data
    df = load_data_from_csv(DATA_PATH)
    df["Recognition_ID"] = (
        df["Dialogue_ID"].astype(str) + "_" + df["Utterance_ID"].astype(str)
    )

    # Group into scenes
    scenes = [
        {
            "dialogue_id": diag_id,
            "utterances": grp[["Utterance", "Speaker", "Recognition_ID", "Emotion"]].to_dict(orient="records"),
        }
        for diag_id, grp in df.groupby("Dialogue_ID")
    ]

    # Resume support
    processed_ids: set = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            if "Dialogue_ID" in existing.columns:
                processed_ids = set(existing["Dialogue_ID"].unique())
                print(f"[Resume] {len(processed_ids)} scenes already done. Skipping.", flush=True)
        except Exception as e:
            print(f"[Resume] Could not read existing file: {e}. Starting fresh.", flush=True)

    unprocessed = [s for s in scenes if s["dialogue_id"] not in processed_ids]
    print(f"[Info] {len(unprocessed)}/{len(scenes)} scenes to process.", flush=True)

    retriever = get_retriever()  # TF-IDF retriever (read-only, thread-safe)
    csv_lock  = threading.Lock()

    def _process_and_save(scene: dict) -> None:
        diag_id = scene["dialogue_id"]
        try:
            scene_results = run_debate_scene(scene, retriever)
        except Exception as exc:
            print(f"\n[ERROR] Scene {diag_id}: {exc}", flush=True)
            return

        if not scene_results:
            return

        with csv_lock:
            result_df   = pd.DataFrame(scene_results)
            file_exists = os.path.exists(OUTPUT_FILE)
            result_df.to_csv(OUTPUT_FILE, mode="a", index=False, header=not file_exists)
            print(f"  [Saved] Scene {diag_id} ({len(scene_results)} rows)", flush=True)

    print(f"[Info] Launching {MAX_SCENE_WORKERS} parallel scene workers.", flush=True)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_SCENE_WORKERS, thread_name_prefix="scene"
    ) as pool:
        futures = [pool.submit(_process_and_save, s) for s in unprocessed]
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures), desc="Scenes", unit="scene",
        ):
            pass

    print("\n[Done] Pipeline complete.", flush=True)
    print(f"       Results: {OUTPUT_FILE}", flush=True)

    # ── Quick eval ────────────────────────────────────────────────────────────
    try:
        from sklearn.metrics import classification_report, f1_score
        final_df = pd.read_csv(OUTPUT_FILE)
        # Deduplicate in case of partial resume overlap
        final_df = final_df.drop_duplicates(subset=["Recognition_ID"], keep="last")

        y_true = final_df["Actual_Emotion"].str.lower().str.strip()
        y_pred = final_df["predicted_emotion"].str.lower().str.strip()
        wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        print("\n" + "=" * 65)
        print(f"  Weighted F1:  {wf1:.4f}  (baseline: 0.7771)")
        print("=" * 65)
        print(classification_report(y_true, y_pred, zero_division=0))

        # Outcome breakdown
        if "outcome" in final_df.columns:
            print("\nOutcome distribution:")
            print(final_df["outcome"].value_counts().to_string())

    except Exception as e:
        print(f"[Eval skipped] {e}", flush=True)


if __name__ == "__main__":
    main()
