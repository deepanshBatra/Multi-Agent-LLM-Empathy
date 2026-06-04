"""
hybrid_pipeline.py
------------------
Hybrid LaERC + InitERC + Selective Agent Pipeline for MELD Emotion Recognition.

Architecture (per utterance):
  [LOCAL]    TF-IDF Retrieval          -> Top-3 MELD training examples (0 API calls)
  [PARALLEL] Empathy Reasoner          -> Micro-linguistic tone tag
             Emotional Shift Detector  -> Turn-to-turn delta signal
  [SEQUENTIAL] LaERC Mental State      -> Dynamic intent + internal state JSON
  [FINAL]    InitERC Classifier        -> Synthesises all signals -> emotion + confidence

Saves incrementally to logs/hybrid_laerc_initerc_results.csv (resume-safe).
3 scenes processed in parallel; global semaphore caps concurrent API calls.
"""

import os
import re
import json
import time
import random
import threading
import concurrent.futures

import pandas as pd
from tqdm import tqdm

import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel

from src.load_data import load_data_from_csv
from src.retrieval_utils import get_retriever

load_dotenv()

# -----------------------------------------------------------------------------
# PARALLELISM & RETRY CONFIGURATION
# -----------------------------------------------------------------------------
# How many scenes to process concurrently (scenes are fully independent).
MAX_SCENE_WORKERS    = 3

# Hard cap on simultaneous in-flight LLM requests across ALL threads.
# With 3 scenes and 2 parallel calls per utterance, ceiling is ~9 -- capped
# here to 5 to stay well within single-endpoint limits.
MAX_CONCURRENT_CALLS = 5

# Retry: exponential backoff (1.5s -> 3s -> 6s -> 12s) + random jitter
MAX_RETRIES          = 4
BASE_BACKOFF_S       = 1.5

# Tokens that indicate a transient / rate-limit error worth retrying
_RETRYABLE_TOKENS = (
    "429", "503", "500", "quota", "resource", "timeout",
    "unavailable", "deadline", "overload", "rate",
)

# Global semaphore -- shared across ALL scene threads
_api_semaphore = threading.Semaphore(MAX_CONCURRENT_CALLS)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "test_sent_emo.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "logs", "hybrid_laerc_initerc_results.csv")
PROMPTS_DIR = os.path.join(BASE_DIR, "src", "llama3_prompts")

MELD_EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]

# -----------------------------------------------------------------------------
# GCP / VERTEX AI SETUP  (mirrors llama_sft_function_calls.py exactly)
# -----------------------------------------------------------------------------
PROJECT_ID  = os.getenv("LLAMA_MODEL_PROJECT_ID") or os.getenv("TUNED_MODEL_PROJECT_ID")
LOCATION    = os.getenv("VERTEX_LOCATION", "us-central1")
ENDPOINT_ID = "2346569469662330880"

vertexai.init(project=PROJECT_ID, location=LOCATION)
llama3_model = GenerativeModel(
    f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
)

# -----------------------------------------------------------------------------
# PROMPT LOADING
# -----------------------------------------------------------------------------
def _load(fname: str) -> str:
    path = os.path.join(PROMPTS_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

PROMPTS = {
    "empathy_reasonar":   _load("empathy_reasonar.txt"),
    "emotional_shift":    _load("emotional_shift.txt"),
    "relational_graph":   _load("relational_graph.txt"),
    "laerc_mental_state": _load("laerc_mental_state.txt"),
    "initerc_classifier": _load("initerc_classifier.txt"),
}

# -----------------------------------------------------------------------------
# LLM CALLER + RETRY WRAPPER
# -----------------------------------------------------------------------------
def _call_llama(system_prompt: str, user_content: str) -> str:
    """
    Formats a Llama 3 chat-template message and calls the SFT endpoint.
    Handles multi-part responses gracefully.
    """
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )
    response = llama3_model.generate_content(prompt)
    try:
        return response.text
    except ValueError:
        return "".join(part.text for part in response.candidates[0].content.parts)


def _call_llama_with_retry(system_prompt: str, user_content: str) -> str:
    """
    Wraps _call_llama with:
      - A global semaphore to cap concurrent in-flight requests.
      - Exponential backoff + jitter for transient errors (429, 503, timeout).
    Raises the original exception if non-retryable or retries are exhausted.
    """
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with _api_semaphore:
                return _call_llama(system_prompt, user_content)
        except Exception as exc:
            last_exc = exc
            err_lower = str(exc).lower()
            is_retryable = any(tok in err_lower for tok in _RETRYABLE_TOKENS)

            if attempt < MAX_RETRIES and is_retryable:
                wait = BASE_BACKOFF_S * (2 ** attempt) + random.uniform(0.1, 1.0)
                print(
                    f"    [Retry {attempt + 1}/{MAX_RETRIES}] {str(exc)[:80]}"
                    f" -- sleeping {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
            else:
                raise  # non-retryable or exhausted

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {last_exc}")


# -----------------------------------------------------------------------------
# JSON / FIELD EXTRACTION  (robust multi-strategy)
# -----------------------------------------------------------------------------
def _extract_json(raw: str) -> dict:
    """
    Attempts multiple strategies to parse a JSON object from raw LLM output.
    Returns a dict (possibly empty) -- never raises.
    """
    raw = str(raw).strip()

    # 1. Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    # 2. Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 3. Grab the LAST {...} block (handles trailing prose)
    matches = list(re.finditer(r"\{[^{}]*\}", raw, re.DOTALL))
    for m in reversed(matches):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue

    # 4. Greedy nested-brace extraction
    start = raw.find("{")
    if start != -1:
        depth, end = 0, -1
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                return json.loads(raw[start: end + 1])
            except json.JSONDecodeError:
                pass

    return {}


def _extract_emotion(raw: str, parsed: dict) -> str:
    """
    Pull predicted_emotion from parsed JSON or fall back to regex on raw text.
    Always returns a canonical MELD label or 'neutral'.
    """
    candidate = str(parsed.get("predicted_emotion", "")).lower().strip()
    if candidate in MELD_EMOTIONS:
        return candidate

    # Regex fallback: look for '"predicted_emotion": "label"'
    m = re.search(r'"?predicted_emotion"?\s*[:\-]\s*"?([a-z]+)"?', raw, re.IGNORECASE)
    if m:
        candidate = m.group(1).lower().strip()
        if candidate in MELD_EMOTIONS:
            return candidate

    # Last resort: scan for any MELD label in the raw text
    pattern = r"\b(" + "|".join(MELD_EMOTIONS) + r")\b"
    m2 = re.search(pattern, raw, re.IGNORECASE)
    if m2:
        return m2.group(1).lower()

    return "neutral"


def _extract_confidence(raw: str, parsed: dict) -> float:
    """Pull confidence float; defaults to 0.5."""
    val = parsed.get("confidence")
    if isinstance(val, (int, float)):
        return float(max(0.0, min(1.0, val)))

    m = re.search(r'"?confidence"?\s*[:\-]\s*([0-9]*\.?[0-9]+)', raw, re.IGNORECASE)
    if m:
        try:
            return float(max(0.0, min(1.0, float(m.group(1)))))
        except ValueError:
            pass

    return 0.5


def _extract_shift_flag(raw: str) -> str:
    """Returns 'TRUE' or 'FALSE' from an Emotional Shift Detector response."""
    if re.search(r"\[SHIFT:\s*TRUE", raw, re.IGNORECASE):
        return "TRUE"
    if re.search(r"\[SHIFT:\s*FALSE", raw, re.IGNORECASE):
        return "FALSE"
    if re.search(r"\bpivot\b|\babrupt\b", raw, re.IGNORECASE):
        return "TRUE"
    return "FALSE"


def _extract_tone_tag(raw: str) -> str:
    """Pull the [LINGUISTIC TONE: ...] tag from Empathy Reasoner output."""
    m = re.search(r"\[LINGUISTIC TONE:\s*([^\]]+)\]", raw, re.IGNORECASE)
    return m.group(1).strip() if m else "UNKNOWN"


# -----------------------------------------------------------------------------
# SPECIALISED AGENT CALLERS  (all use _call_llama_with_retry)
# -----------------------------------------------------------------------------
def call_empathy_reasoner(utterance: str) -> str:
    return _call_llama_with_retry(PROMPTS["empathy_reasonar"], utterance)


def call_emotional_shift(
    prev_utt: str, prev_spk: str,
    target_utt: str, target_spk: str,
    context_summary: str,
) -> str:
    user_content = (
        f"CONTEXT SUMMARY:\n{context_summary}\n\n"
        f"PREVIOUS UTTERANCE:\nSpeaker: {prev_spk}\nText: {prev_utt}\n\n"
        f"TARGET UTTERANCE:\nSpeaker: {target_spk}\nText: {target_utt}"
    )
    return _call_llama_with_retry(PROMPTS["emotional_shift"], user_content)


def call_relational_graph(scene_script: str) -> str:
    return _call_llama_with_retry(
        PROMPTS["relational_graph"],
        f"Scene Dialogue:\n{scene_script}",
    )


def call_laerc_mental_state(
    utterance: str,
    speaker: str,
    history_turns: list,
) -> str:
    """
    Fills the LaERC prompt template and calls the model.
    history_turns: list of {'Speaker': ..., 'Utterance': ...} dicts (last 5).
    """
    if history_turns:
        history_text = "\n".join(
            f"{t['Speaker']}: {t['Utterance']}" for t in history_turns
        )
    else:
        history_text = "[No prior context -- first utterance in scene]"

    system_prompt = (
        PROMPTS["laerc_mental_state"]
        .replace("{HISTORY}", history_text)
        .replace("{SPEAKER}", speaker)
        .replace("{UTTERANCE}", utterance)
    )
    return _call_llama_with_retry(system_prompt, "Provide the JSON output now.")


def call_initerc_classifier(
    recognition_id: str,
    utterance: str,
    speaker: str,
    short_history: list,
    retrieved_examples_text: str,
    linguistic_signal: str,
    shift_signal: str,
    mental_state_raw: str,
    relational_signal: str,
) -> str:
    """Builds the final InitERC prompt and calls the model."""
    short_hist_text = (
        "\n".join(f"{t['Speaker']}: {t['Utterance']}" for t in short_history)
        if short_history
        else "[Start of scene]"
    )

    user_content = (
        PROMPTS["initerc_classifier"]
        .replace("{RECOGNITION_ID}", str(recognition_id))
        .replace("{SPEAKER}", str(speaker))
        .replace("{UTTERANCE}", str(utterance))
        .replace("{SHORT_HISTORY}", short_hist_text)
        .replace("{RETRIEVED_EXAMPLES}", retrieved_examples_text)
        .replace("{LINGUISTIC_SIGNAL}", linguistic_signal)
        .replace("{SHIFT_SIGNAL}", shift_signal)
        .replace("{MENTAL_STATE}", mental_state_raw)
        .replace("{RELATIONAL_SIGNAL}", relational_signal)
    )
    return _call_llama_with_retry(
        "You are an expert emotion recognition system. Follow the protocol exactly.",
        user_content,
    )


# -----------------------------------------------------------------------------
# SCENE-LEVEL PIPELINE
# -----------------------------------------------------------------------------
def run_hybrid_scene(scene_obj: dict, retriever) -> list:
    utterances  = scene_obj["utterances"]
    dialogue_id = scene_obj["dialogue_id"]

    # Precompute scene-level relational graph (1 call per scene, not per utterance)
    scene_script = "\n".join(
        f"{u['Speaker']}: {u['Utterance']}" for u in utterances
    )
    print(f"  [RelGraph] Scene {dialogue_id}...", flush=True)
    relational_signal = call_relational_graph(scene_script)

    context_summary = f"Scene {dialogue_id} -- {len(utterances)} turns."

    results = []

    for idx, u in enumerate(utterances):
        target_utt = u.get("Utterance", "")
        speaker    = u.get("Speaker", "Unknown")
        rec_id     = u.get("Recognition_ID", f"{dialogue_id}_{idx}")
        actual_emo = u.get("Emotion", "unknown")

        history_last5 = utterances[max(0, idx - 5): idx]
        history_last3 = utterances[max(0, idx - 3): idx]
        prev          = utterances[idx - 1] if idx > 0 else None

        print(f"  [Utterance {idx + 1}/{len(utterances)}] {rec_id}", flush=True)

        # STAGE 0: Local TF-IDF retrieval (no API call)
        retrieved     = retriever.get_top_k_examples(target_utt, k=3)
        retrieved_txt = retriever.format_for_prompt(retrieved)

        # STAGE 1: Parallel -- Empathy Reasoner + Shift Detector
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            f_ling  = ex.submit(call_empathy_reasoner, target_utt)
            f_shift = ex.submit(
                call_emotional_shift,
                prev["Utterance"] if prev else "",
                prev["Speaker"]   if prev else "Unknown",
                target_utt,
                speaker,
                context_summary,
            )
            linguistic_signal = f_ling.result()
            shift_raw         = f_shift.result()

        shift_flag = _extract_shift_flag(shift_raw)

        # STAGE 2: LaERC Mental State
        mental_state_raw = call_laerc_mental_state(target_utt, speaker, history_last5)

        # STAGE 3: InitERC Final Classification
        raw_final = call_initerc_classifier(
            recognition_id=rec_id,
            utterance=target_utt,
            speaker=speaker,
            short_history=history_last3,
            retrieved_examples_text=retrieved_txt,
            linguistic_signal=linguistic_signal,
            shift_signal=f"[SHIFT: {shift_flag}] -- {shift_raw[:300]}",
            mental_state_raw=mental_state_raw,
            relational_signal=relational_signal,
        )

        # Parse output
        parsed        = _extract_json(raw_final)
        predicted_emo = _extract_emotion(raw_final, parsed)
        confidence    = _extract_confidence(raw_final, parsed)
        reasoning     = parsed.get("reasoning", "")

        print(
            f"    pred={predicted_emo} (conf={confidence:.2f}) | actual={actual_emo}",
            flush=True,
        )

        results.append({
            "Dialogue_ID":           dialogue_id,
            "Recognition_ID":        rec_id,
            "Speaker":               speaker,
            "Utterance":             target_utt,
            "Actual_Emotion":        actual_emo,
            "predicted_emotion":     predicted_emo,
            "confidence":            confidence,
            "reasoning":             reasoning,
            "shift_flag":            shift_flag,
            "mental_state_raw":      mental_state_raw[:300],
            "linguistic_signal_tag": _extract_tone_tag(linguistic_signal),
        })

    return results


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 65, flush=True)
    print("  Hybrid LaERC + InitERC Pipeline  --  MELD Test Set", flush=True)
    print(f"  Output -> {OUTPUT_FILE}", flush=True)
    print(
        f"  Workers: {MAX_SCENE_WORKERS} scenes | "
        f"Semaphore: {MAX_CONCURRENT_CALLS} concurrent calls | "
        f"Retries: {MAX_RETRIES}",
        flush=True,
    )
    print("=" * 65, flush=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load test data
    df = load_data_from_csv(DATA_PATH)
    df["Recognition_ID"] = (
        df["Dialogue_ID"].astype(str) + "_" + df["Utterance_ID"].astype(str)
    )

    # Group into scenes
    scenes = []
    for diag_id, grp in df.groupby("Dialogue_ID"):
        scenes.append({
            "dialogue_id": diag_id,
            "utterances": grp[
                ["Utterance", "Speaker", "Recognition_ID", "Emotion"]
            ].to_dict(orient="records"),
        })

    # Resume support -- skip already-saved Dialogue_IDs
    processed_ids: set = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            if "Dialogue_ID" in existing.columns:
                processed_ids = set(existing["Dialogue_ID"].unique())
                print(
                    f"[Resume] {len(processed_ids)} scenes already done. Skipping.",
                    flush=True,
                )
        except Exception as e:
            print(f"[Resume] Could not read existing file: {e}. Starting fresh.", flush=True)

    unprocessed = [s for s in scenes if s["dialogue_id"] not in processed_ids]
    print(f"[Info] {len(unprocessed)}/{len(scenes)} scenes to process.", flush=True)

    # Build retriever once -- shared safely across all threads (read-only after init)
    retriever = get_retriever()

    # Thread-safe CSV writer
    csv_lock = threading.Lock()

    def _process_and_save(scene: dict) -> None:
        """Worker: run one scene and immediately persist results to CSV."""
        diag_id = scene["dialogue_id"]
        try:
            scene_results = run_hybrid_scene(scene, retriever)
        except Exception as exc:
            print(f"\n[ERROR] Scene {diag_id}: {exc}", flush=True)
            return

        if not scene_results:
            return

        with csv_lock:
            result_df   = pd.DataFrame(scene_results)
            file_exists = os.path.exists(OUTPUT_FILE)
            result_df.to_csv(
                OUTPUT_FILE,
                mode="a",
                index=False,
                header=not file_exists,
            )
            print(
                f"  [Saved] Scene {diag_id} ({len(scene_results)} rows) -> {OUTPUT_FILE}",
                flush=True,
            )

    print(
        f"[Info] Launching {MAX_SCENE_WORKERS} parallel scene workers "
        f"(semaphore cap={MAX_CONCURRENT_CALLS} concurrent API calls).",
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_SCENE_WORKERS,
        thread_name_prefix="scene",
    ) as scene_pool:
        futures = [scene_pool.submit(_process_and_save, scene) for scene in unprocessed]
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Scenes",
            unit="scene",
        ):
            pass  # saving is done inside _process_and_save

    print("\n[Done] Pipeline complete.", flush=True)
    print(f"       Results at: {OUTPUT_FILE}", flush=True)

    # Quick eval
    try:
        from sklearn.metrics import classification_report, f1_score
        final_df = pd.read_csv(OUTPUT_FILE)
        y_true = final_df["Actual_Emotion"].str.lower().str.strip()
        y_pred = final_df["predicted_emotion"].str.lower().str.strip()
        wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print("\n" + "=" * 65)
        print(f"  Weighted F1: {wf1:.4f}  (target > 0.7891)")
        print("=" * 65)
        print(classification_report(y_true, y_pred, zero_division=0))
    except Exception as e:
        print(f"[Eval skipped] {e}", flush=True)


if __name__ == "__main__":
    main()
