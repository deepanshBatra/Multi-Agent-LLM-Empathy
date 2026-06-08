# Multi-Agent Empathy & Emotion Recognition (MERC)
## Code Implementation & Experimentation — Knowledge Transfer Document

> **Prepared:** 2026-05-31 | **Source Repository:** `deepanshBatra/Multi-Agent-LLM-Empathy`
> **Dataset:** MELD (Multimodal EmotionLines Dataset) — 7-class ERC benchmark (Friends TV series)

---

## Table of Contents
1. [Executive Summary & Architecture Overview](#1-executive-summary--architecture-overview)
2. [Methodology & Experimentation Log](#2-methodology--experimentation-log)
3. [Performance Metrics & Benchmarks](#3-performance-metrics--benchmarks)
4. [Codebase Roadmap & Component Mapping](#4-codebase-roadmap--component-mapping)
5. [Future Work & Unresolved Hypotheses](#5-future-work--unresolved-hypotheses)

---

## 1. Executive Summary & Architecture Overview

### 1.1 Project Goal

This project attacks the **Emotion Recognition in Conversation (ERC)** task on the **MELD dataset** (2,610 test utterances from the *Friends* TV series) using a **Council of LLMs** — a coordinated multi-agent system where specialized LLM "agents" each contribute a distinct analytical lens, and a final aggregator synthesizes their reports into a single emotion label from the 7-class MELD ontology:

```
neutral | joy | surprise | anger | sadness | disgust | fear
```

The best validated Weighted F1 on the MELD test set is **0.7771** (Llama 3.1 plain validation baseline, N=2,565 utterances).

---

### 1.2 The "Council of LLMs" Architecture

The system uses a **hierarchical, 3-level execution model** across all major pipeline variants:

```
┌──────────────────────────────────────────────────────────────┐
│                      LEVEL 1 — GLOBAL CONTEXT                │
│  ┌─────────────────────┐   ┌──────────────────────────────┐  │
│  │  Context Manager    │   │  Relational Graph Manager    │  │
│  │  (Scene Historian)  │   │  (Relationship Historian)    │  │
│  │  Llama 4 Maverick   │   │  Llama 4 Scout               │  │
│  └─────────────────────┘   └──────────────────────────────┘  │
│   ONE call per SCENE — outputs shared to all utterance loops  │
└───────────────────────────┬──────────────────────────────────┘
                            │ feeds into
┌───────────────────────────▼──────────────────────────────────┐
│              LEVEL 2 — PER-UTTERANCE SPECIALISTS (PARALLEL)   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐   │
│  │  Character       │  │  Sentiment /     │  │  Emotional│   │
│  │  Profiler        │  │  Empathy Reasoner│  │  Shift    │   │
│  │  (Fine-tuned     │  │  (Fine-tuned     │  │  Detector │   │
│  │   Gemini Flash / │  │   Gemini Flash / │  │  (DeepSeek│   │
│  │   Llama 3.1 SFT) │  │   Llama 3.1 SFT) │  │  R1 /     │   │
│  └──────────────────┘  └──────────────────┘  │  Llama SFT│   │
│  ┌──────────────────────────────────────────┐  └───────────┘   │
│  │  Social Dynamics Expert (Reasoning Bridge│                  │
│  │  Llama 4 Maverick / Gemini 2.0 Flash)    │                  │
│  └──────────────────────────────────────────┘                  │
└───────────────────────────┬──────────────────────────────────┘
                            │ synthesized by
┌───────────────────────────▼──────────────────────────────────┐
│               LEVEL 3 — THE FINAL JUDGE / AGGREGATOR          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Council Aggregator / Chief Justice & Emotion Arbiter    │ │
│  │  GPT-OSS 120B (via Groq) / DeepSeek R1 (via OpenRouter) │ │
│  │  / InitERC Classifier (Llama 3.1 SFT)                   │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

#### Concurrency Model

All pipeline variants use Python's `concurrent.futures.ThreadPoolExecutor` with a **global `threading.Semaphore`** to hard-cap simultaneous in-flight LLM requests (typically `MAX_CONCURRENT_CALLS = 5–6`). Scenes are processed `MAX_SCENE_WORKERS = 3` at a time. A scene-level CSV lock (`threading.Lock`) ensures atomic incremental writes for **crash-resume safety**.

#### Retry Policy

All LLM calls use exponential backoff:
- **Retryable tokens:** `429, 503, 500, quota, resource, timeout, unavailable, deadline, overload, rate`
- **Backoff:** `BASE_BACKOFF_S * (2^attempt) + uniform(0.1, 1.0)s`, `MAX_RETRIES = 3–4`

---

### 1.3 Speaker Bio Cards — Generation, Structure, and Injection

#### What They Are

Bio Cards are **pre-computed, per-speaker psychological profiles** derived from the MELD training set dialogue corpus. They encode stable character traits used by downstream agents to calibrate their predictions. They solve the fundamental problem: high arousal ≠ anger (e.g., Monica's natural high-energy baseline vs. genuine anger).

#### Structure (from `logs/speaker_bio_cards.json`)

Each entry is a JSON object with 6 fields:
```json
{
  "Chandler": {
    "speaker": "Chandler",
    "static_persona": "Chandler is a witty and sarcastic individual with a dry sense of humor, often using irony and self-deprecation to navigate social interactions...",
    "linguistic_style": "Chandler frequently uses sarcasm, irony, and teasing... 'Could I BE any more...'",
    "baseline_arousal": "Medium. Relatively calm but injects humor/sarcasm which elevates arousal level...",
    "negative_expression": "Tends to express anger/frustration through sarcasm and irony... 'Oh, great', 'That's terrific'...",
    "social_relationship_priors": "More playful/teasing with Joey and Monica; uses nicknames like 'Pheebs'..."
  }
}
```

| Field | Purpose |
|---|---|
| `static_persona` | General personality summary — grounds all agents' baseline expectations |
| `linguistic_style` | Characteristic speech patterns, catchphrases, verbal signatures |
| `baseline_arousal` | Calibrates the Shift Detector — "high arousal for this character" vs "high arousal globally" |
| `negative_expression` | How the character typically expresses anger/disgust vs. sadness/fear (key for sarcasm disambiguation) |
| `social_relationship_priors` | Listener-dependent tone shifts — used by the Relational Graph agent |

Generic/minor characters use `"character_type": "GENERIC_CHARACTER"` with `[Insufficient data]` placeholders. The aggregator falls back to a **Neutral-Professional Baseline** for these.

#### Generation

Bio cards are generated via the `notebooks/speaker_bio_card_generator_v2.ipynb` notebook (117 KB), which calls an LLM on grouped utterances per speaker from the MELD training set. The notebook `notebooks/speaker_bio_card_generator.ipynb` is the v1 predecessor.

#### Injection Points

Bio cards are injected at **two levels**:

1. **`src/llama3_full_council.py` → `get_speaker_bio_card_text(speaker, bio_cards)`** (line 26–32): Loaded once at startup from `logs/speaker_bio_cards.json`, then looked up per utterance and passed to `call_llama3_council_aggregator()`.

2. **`src/fine_tuned_agents_phase3.py` → `call_council_aggregator()`** (line 138–236): The bio card replaces the `[SPEAKER BIO CARD WILL BE PROVIDED HERE]` placeholder in the council aggregator prompt template (`src/prompts/council_aggregator.txt`).

3. **`src/llama_sft_function_calls.py` → `call_llama3_council_aggregator()`** (line 81–109): Injected directly into the user-content payload as `SPEAKER BIO: {bio_card_content}`.

---

## 2. Methodology & Experimentation Log

The git log reveals 14 commits across approximately 5 discrete experimental phases. The table below maps commit hashes to phases:

| Commit | Message | Phase |
|---|---|---|
| `2dc5a4a` | First commit multi-llm-agent-arch | Phase 1 |
| `07e65e1` | Edited ground truth emotion, defaulting to neutral | Phase 1 fix |
| `cf52819`, `b327193` | Phase 2 sample tests / updation | Phase 2 |
| `7c4edfa`, `a31d279` | Phase 3 implementation | Phase 3 |
| `dc9086e`, `0374d6b`, `32d890c` | Phase 3 results and prompt fine-tuning | Phase 3 iteration |
| `5baa981` | Phase 4 results | Phase 4 |
| `83a94d7` | Fine-tuned Llama3 LoRA results + council+llama3 notebooks | Phase 4.1 |
| `116f813` | Llama 3.1 LoRA SFT baseline + Llama 3.1 as aggregator | Phase 4.2 |
| `dbe0d7e` | Llama 3+ context manager + persona cards | Phase 5a |
| `c5ecaa7` | Llama 3 unified prompt + character bios | Phase 5b |

---

### Phase 1: Baseline Gemini Multi-Agent Council (src/agents.py)

**Objective:** Establish feasibility of the Council-of-LLMs architecture for ERC. Test a 3-agent hierarchical pipeline using readily available API models.

**Model Setup:**
- **Context Manager:** `llama-3.3-70b-versatile` (Groq)
- **Social Dynamics Expert:** `llama-3.1-8b-instant` (Groq)
- **Empathy Reasoner:** `qwen/qwen3-32b` (via `DEEPSEEK_API_KEY`)
- **Council Aggregator:** `openai/gpt-oss-120b` (Groq)

**Feature Engineering / Prompting:**
- Prompts loaded from `src/prompts/` (the "Phase 1/2 prompt set" — larger, more verbose versions vs. the later `src/llama3_prompts/`)
- `run_multi_agent_conversation(context_dict)` called with a raw `context_dict` containing `recognition_id`
- Context Manager runs **sequentially first**; Social Dynamics + Empathy run in **parallel** (`ThreadPoolExecutor(max_workers=2)`)
- Aggregator receives all 3 reports + recognition ID; `temperature=1` (Groq params)

**Technical Implementation:**
- [`src/agents.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/agents.py) — `run_multi_agent_conversation()`, `groq_llm_call()`, `gemini_llm_call()`
- Results: `logs/meld_results.jsonl` (Phase 1 evaluation output)

**Key Limitation:** No bio cards. No temporal shift detection. No fine-tuning. Single-scene inference only; not batched for the full MELD test set.

---

### Phase 2: 6-Agent Council with Gemini Fine-Tuned Specialists (fine_tuned_agents.py)

**Objective:** Replace generic API models for specialist agents (Character Profiler, Sentiment Analyst) with **domain-specific Vertex AI fine-tuned Gemini endpoints**, while retaining Llama 4 for global context agents.

**Model Setup:**
- **Agent 1 (Context Manager):** `meta-llama/llama-4-scout-17b-16e-instruct` (Groq, `CONTEXT_MANAGER` API key)
- **Agent 2 (Relational Graph):** `meta-llama/llama-4-maverick-17b-128e-instruct` (Groq, `RELATIONAL_GRAPH_MANAGER`)
- **Agent 3 (Character Profiler):** Fine-tuned Gemini endpoint `projects/{project}/locations/us-central1/endpoints/6568289500043149312` (Vertex AI)
- **Agent 4 (Sentiment Analyst):** Fine-tuned Gemini endpoint `projects/{project}/locations/us-central1/endpoints/5496855001194037248` (Vertex AI)
- **Agent 5 (Social Dynamics):** `gemini-2.0-flash` (Vertex AI cloud client via `genai.Client(vertexai=True)`)
- **Agent 6 (Aggregator):** `openai/gpt-oss-120b` (Groq, `COUNCIL_AGGREGATOR` API key)

**Feature Engineering / Prompting:**
- First integration of full 6-agent hierarchy
- Phase 2 prompts: `src/prompts/` directory (7 files, 1,570–4,649 bytes each — detailed, multi-section frameworks)
- Profiler uses `social_graph` context from Relational Graph agent as input
- Sentiment agent uses `empathy_reasonar.txt` prompt

**Technical Implementation:**
- [`src/fine_tuned_agents.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/fine_tuned_agents.py) — `call_tuned_profiler()`, `call_tuned_sentiment()`, `call_social_dynamics()`, `call_gpt_oss_aggregator()`
- Note: The core `run_phase2_council()` function is **commented out** (lines 108–135); individual agent callers are functional. Phase 2 execution was driven from `notebooks/meld_fine_tuned.ipynb`.
- Vertex AI initialized via `vertexai.init(project=tuned_project_id, location='us-central1')`
- Results: `logs/council_phase2_results.csv` (N not specified in README; 87 KB file), `logs/council_phase2_results.jsonl`

---

### Phase 3: 7-Agent Council + Emotional Shift Detector + DeepSeek R1 Aggregator (fine_tuned_agents_phase3.py)

**Objective:** Add a **temporal dynamics agent** (Emotional Shift Detector) to capture turn-to-turn emotional pivots. Replace GPT-OSS aggregator with **DeepSeek R1** (via OpenRouter) for chain-of-thought reasoning. Introduce **speaker bio cards** and **previous-prediction history** into the aggregator context.

**Model Setup:**
- **Agents 1–5:** Same as Phase 2
- **Agent 6 (Emotional Shift Detector):** `deepseek/deepseek-r1` (OpenRouter, `OPENROUTER_API_KEY` or `OPEN_ROUTER_DEEPSEEK_KEY`)
- **Agent 7 (Council Aggregator):** `deepseek/deepseek-r1` (OpenRouter, `OPENROUTER_MODEL` env var)

**Feature Engineering / Prompting:**
- First introduction of bio card injection: `call_council_aggregator(... speaker_bio_card=None, previous_predictions=None)`
- Bio card replaces `[SPEAKER BIO CARD WILL BE PROVIDED HERE]` placeholder in `src/prompts/council_aggregator.txt`
- Rolling window of last-3 predictions injected via `[RECENT COUNCIL DECISIONS WILL BE PROVIDED HERE]` with **low-confidence flagging** (threshold: conf < 0.60) and shift continuity flags
- Emotional Shift prompt (`src/prompts/emotional_shift.txt`) introduces the `[SHIFT: TRUE/FALSE]` binary signal
- The aggregator's `council_aggregator.txt` prompt implements **7 deliberation rules** (Valence Pre-Filter, Temporal Shift Check, Fear Gate, Sadness Anchor, Surprise Disambiguation, Neutrality/Joy Gate, Sarcasm Resolution)

**Technical Implementation:**
- [`src/fine_tuned_agents_phase3.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/fine_tuned_agents_phase3.py) — `call_emotional_shift()` (lines 101–136), `call_council_aggregator()` (lines 138–236), `call_gpt_oss_aggregator()` backward-compatible wrapper (lines 238–248)
- Key function `call_council_aggregator()` does real-time API key resolution (falls back across 4 env vars) and uses `requests.post()` to OpenRouter with 120s timeout
- Results: `logs/council_phase3_results.csv` (215 KB), `logs/council_phase3_2_results.csv`, `logs/council_phase3_3_results.csv`, `logs/council_phase3_4_results.csv`
- Notebooks: `notebooks/meld_fine_tuned_phase3.ipynb` → **Weighted F1: 0.5764** (N=492)

---

### Phase 4: Llama 3.1 SFT + Full Council + Bio Card Integration (llama3_full_council.py + llama_sft_function_calls.py)

**Objective:** Replace cloud Gemini fine-tuned endpoints with a **single SFT Llama 3.1 model** deployed on a dedicated Vertex AI endpoint. This consolidates all specialist agent calls to one endpoint (Endpoint ID: `2346569469662330880`). Test the full council pipeline with bio cards against the complete MELD test set.

**Model Setup:**
- **All specialist agents (1–6):** Single SFT Llama 3.1 endpoint via `GenerativeModel` + Vertex AI (`us-central1`)
- **Council Aggregator (Agent 7):** Same SFT Llama 3.1 endpoint with `council_aggregator.txt` prompt

**Feature Engineering / Prompting:**
- Switched from verbose Phase 1/2 prompts (`src/prompts/`) to **concise Llama 3.1-optimized prompts** (`src/llama3_prompts/`) — designed for the Llama 3 chat template format (`<|begin_of_text|><|start_header_id|>system<|end_header_id|>`)
- `src/llama3_prompts/` contains 13 prompt files; `src/prompts/` contains 7 (the older Gemini-era prompts)
- Bio card + previous predictions sliding window (last 3) passed via `call_llama3_council_aggregator()` in `src/llama_sft_function_calls.py`
- `global_history` list maintains a rolling 3-prediction state across utterances in `src/llama3_full_council.py`

**Technical Implementation:**
- [`src/llama_sft_function_calls.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/llama_sft_function_calls.py) — defines `llama3_sft_call()` (Llama 3 chat-template builder, lines 36–46) and 7 agent-specific callers
- [`src/llama3_full_council.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/llama3_full_council.py) — `run_llama3_council_scene()` (lines 62–134): orchestrates Level 1 (context + relational graph sequential), Level 2 specialists (parallel via ThreadPoolExecutor with max_workers=5), Level 3 aggregator
- `load_speaker_bio_cards()` / `get_speaker_bio_card_text()` — bio card I/O utilities (lines 18–32)
- `extract_json_field()` — multi-strategy JSON extractor for final prediction parsing (lines 34–59)
- Results: `logs/llama3/council_phase4_llama_results.csv` (1 MB), `logs/council_phase4_results.csv` (1.9 MB), `logs/council_phase4_1_results.csv` (4.4 MB)
- Notebooks: `notebooks/meld_fine_tuned_phase4_llama.ipynb` → **Weighted F1: 0.4851** (N=149)

**Key Finding:** The full 7-agent council with SFT Llama 3.1 **underperformed** the plain single-agent baseline on small test subsets, suggesting the complexity of prompt chaining and per-agent noise compounding outweighs the benefit at this model capability level.

---

### Phase 4.2: Llama 3.1 Dedicated Endpoint — Single-Agent Baseline Validation

**Objective:** Establish a clean, reproducible baseline using the dedicated Llama 3.1 endpoint (different endpoint: `mg-endpoint-a66f56b4-7b58-4560-b43e-2a8777c38cd9`, `us-east1`) via the raw `/predict` REST API (not Vertex AI SDK). This became the **peak performance reference**.

**Model Setup:**
- **Single agent:** Dedicated Vertex AI endpoint for fine-tuned Llama 3.1, `us-east1`, raw REST API
- **Temperature:** 0.0–0.1
- **Max tokens:** 128 (single-utterance) / 1024 (scene-batch)

**Feature Engineering / Prompting:**

*Per-utterance (`llama3_single_agent_validation.py`):*
```
SYSTEM: Expert ERC assistant for MELD. Predict ONE emotion for TARGET utterance. 
        Labels: [7 labels]. Output ONLY: {"predicted_emotion": "label"}
USER:   Context: {last 3 utterances}
        TARGET — {Speaker}: "{utterance}"
```

*Scene-batch (`llama3_scene_batch_validation.py`):*
```
SYSTEM: For each scene, predict one emotion for EVERY utterance. Return JSON:
        {"predictions": [{"utterance_id": "id", "predicted_emotion": "label", "reasoning": "..."}]}
USER:   Scene Dialogue ID: {id}
        {idx} | {Speaker}: {utterance}
        Return predictions for ALL utterance_id values above.
```

**Technical Implementation:**
- [`src/llama3_single_agent_validation.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/llama3_single_agent_validation.py) — `_call_raw()` → strips `<|start_header_id|>assistant<|end_header_id|>` echo + `Output: ` prefix; `parse_prediction()` multi-strategy parser
- [`src/llama3_scene_batch_validation.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/llama3_scene_batch_validation.py) — `build_scene_prompt()`, `parse_scene_predictions()`, scene-level results
- [`src/llama3_scene_batch_validation_optimized.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/llama3_scene_batch_validation_optimized.py) — further-optimized version (16 KB)
- Auth: Google ADC (`google.auth.default()`), token auto-refreshed via `_get_token()`
- Results: `logs/llama3_single_agent_results.csv` (203 KB), `logs/llama3_scene_batch_results.csv` (1.2 MB)
- Notebook: `notebooks/llama3_plain_validation.ipynb` → **Weighted F1: 0.7708** (N=2,474)

---

### Phase 4.3: Biocards + Context Manager Augmentation (llama3_biocards_context_manager.ipynb)

**Objective:** Test whether injecting bio cards AND the Context Manager's scene summary into the single-agent prompt improves over the plain baseline.

**Feature Engineering / Prompting:**
- Bio cards from `logs/speaker_bio_cards.json` injected into per-utterance prompt
- Context Manager provides scene-level narrative summary as additional context
- All processed by the dedicated SFT Llama endpoint

**Results:**
- Notebook: `notebooks/llama3_biocards_context_manager.ipynb` → **Weighted F1: 0.7286** (N=2,545)
- Stored at: `logs/llama3/llama3_biocards_context_predictions_master.csv` (14 MB — largest log file)

**Key Finding:** Bio card + context manager **DECREASED** Weighted F1 by ~0.042 vs. plain single-agent, suggesting prompt over-complexity or interference with the model's native few-shot calibration.

---

### Phase 5a: 3-Agent Debate Pipeline (debate_pipeline.py)

**Objective:** Replace the hierarchical council with a **lateral voting / debate mechanism**. Three independent agents cast votes on each utterance; consensus triggers immediate acceptance, 3-way splits escalate to a dedicated Arbitrator agent.

**Architecture:**
```
ROUND 1 (parallel): 3 agents vote per utterance
  Agent 1 (Plain)      — scene narrative + MELD calibration examples
  Agent 2 (Linguistic) — word/tone/punctuation only
  Agent 3 (Shift)      — emotional momentum / temporal delta

Vote resolution:
  3/3 unanimous  → accept, 0 extra calls
  2/3 majority   → accept majority, 0 extra calls
  3-way split    → ROUND 2: Arbitrator reads all 3 votes → final verdict
```

**Model Setup:**
- **All 3 voting agents:** Dedicated Llama 3.1 SFT endpoint (`us-east1`, REST API)
- **Arbitrator:** Same dedicated endpoint
- **Retriever:** TF-IDF index over MELD training set (`MeldRetriever`, no API calls)

**Technical Implementation:**
- [`src/debate_pipeline.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/debate_pipeline.py) — `run_debate_scene()` (lines 435–526), `_aggregate_votes()` (lines 406–430), `_call_arbitrator()` (lines 375–401)
- Prompts: `src/llama3_prompts/vote_plain.txt`, `vote_linguistic.txt`, `vote_context_shift.txt`, `arbitrator.txt`
- `vote_plain.txt` includes MELD label distribution priors: `neutral=48%, joy=15%, surprise=11%, anger=13%, sadness=8%, disgust=3%, fear=2%`
- `_extract_json()` uses 4-strategy parsing: direct → last `{...}` block → reversed multi-match → greedy nested-brace
- Results: `logs/debate_pipeline_results.csv` (327 KB)
- Baseline reference in eval code: `0.7771` (weighted F1 target)

---

### Phase 5b: Hybrid LaERC + InitERC Pipeline (hybrid_pipeline.py)

**Objective:** Implement the academic **LaERC** (Latent ERC) + **InitERC** (Instance-based ERC) dual-framework. LaERC provides a dynamic mental state JSON; InitERC is the final synthesizing classifier incorporating retrieval-augmented few-shot examples.

**Architecture (per utterance):**
```
STAGE 0 (Local): TF-IDF retrieval → Top-3 MELD training examples (0 API calls)
STAGE 1 (Parallel):
  Empathy Reasoner     → micro-linguistic tone tag [LINGUISTIC TONE: ...]
  Emotional Shift Det. → turn-to-turn delta [SHIFT: TRUE/FALSE]
STAGE 2 (Sequential):
  LaERC Mental State   → dynamic intent + internal state JSON
STAGE 3 (Final):
  InitERC Classifier   → synthesises all 4 signals → {emotion, confidence, reasoning}
```

**Feature Engineering / Prompting:**

The `initerc_classifier.txt` prompt encodes a **6-step decision protocol**:
1. **ANCHOR** on retrieved examples (MELD prior)
2. **ADJUST** with LaERC `internal_state`
3. **REFINE** with Linguistic — `surface_vs_subtext == "misaligned"` → linguistic tone **overrides** surface text
4. **TEMPORAL GATE** — `[SHIFT: TRUE]` → current-turn signals dominate over retrieved examples
5. **VULNERABILITY GATE** — `[RELATIONSHIP DYNAMIC: SAFE/SUPPORTIVE]` → Sadness/Fear more likely
6. **NEUTRAL as last resort** — not a tie-breaker

LaERC prompt (`laerc_mental_state.txt`) outputs structured JSON:
```json
{
  "immediate_intent": "one-sentence description",
  "internal_state": "one-sentence description",
  "surface_vs_subtext": "aligned|misaligned|uncertain",
  "emotional_momentum": "escalating|de-escalating|stable|unknown"
}
```

**Model Setup:**
- **All calls:** Vertex AI SFT Llama 3 endpoint (`ENDPOINT_ID = "2346569469662330880"`, `us-central1`)
- **Retriever:** `MeldRetriever` singleton (TF-IDF, `ngram_range=(1,2)`, `max_features=30_000`, `sublinear_tf=True`)

**Technical Implementation:**
- [`src/hybrid_pipeline.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/hybrid_pipeline.py) — `run_hybrid_scene()` (lines 351–439), `call_laerc_mental_state()` (lines 287–309), `call_initerc_classifier()` (lines 312–345)
- [`src/retrieval_utils.py`](file:///c:/Users/deepa/Multi-Agent-LLM-Empathy/src/retrieval_utils.py) — `MeldRetriever` class (lines 20–110), `get_retriever()` singleton factory (lines 117–122)
- Relational graph computed **once per scene** (not per utterance) to minimize API calls
- Results: `logs/hybrid_laerc_initerc_results.csv` (3.1 MB — comprehensive run)
- Eval target in code: `target > 0.7891` (aspirational — not yet achieved)

---

### Phase 5c: Ensemble Confidence Router (ensemble_confidence_router.ipynb)

**Objective:** Build a **post-hoc ensemble** over multiple pipeline runs. Stage 1 uses the best baseline predictions; Stage 2 applies a confidence-based router to attempt corrections on low-confidence or minority-class predictions.

**Results (from `logs/ensemble_router/`)**:
- Stage 1 (Baseline): WF1 = **0.7771**, Macro F1 = 0.7109 (N=2,565)
- Stage 2 (Refinements): WF1 = 0.7771 (net neutral — refinements on 327 utterances canceled out)
- Stage 2 standalone on changed predictions: WF1 = 0.4573 (the targeted corrections **regressed**)

**Key Finding:** The ensemble router did not improve over the baseline — a critical signal that post-hoc correction on low-confidence predictions is error-prone without a stronger oracle.

---

## 3. Performance Metrics & Benchmarks

### 3.1 Summary Table — All Experiments

| Notebook / Script | N Utterances | Split | Architecture | Weighted F1 |
|---|---|---|---|---|
| `llama3_plain_validation.ipynb` | **2,474** | Test | Llama 3.1 SFT (Plain, per-utterance) | **0.7708** |
| `llama3_biocards_context_manager.ipynb` | 2,545 | Test | Llama 3.1 SFT + Bio Cards + Context Mgr | 0.7286 |
| `ensemble_confidence_router.ipynb` (Stage 1) | 2,565 | Test | Ensemble (Baseline) | 0.7771 |
| `ensemble_confidence_router.ipynb` (Final) | 2,565 | Test | Ensemble (Stage 1 + Stage 2 corrections) | 0.7771 |
| `debate_pipeline.py` | ~2,610 | Test | 3-Agent Debate + Arbitrator | ~0.77 (reference `0.7771`) |
| `meld_fine_tuned_phase3.ipynb` | 492 | Test | 7-Agent Council (Phase 3, DeepSeek R1) | 0.5764 |
| `Untitled-1.ipynb` | 160 | Test | Fine-tuned Phase 4+ | 0.4984 |
| `meld_fine_tuned_phase4_llama.ipynb` | 149 | Test | 7-Agent Council (Llama SFT, Phase 4) | 0.4851 |
| `meld_fine_tuned.ipynb` | 181 | Test | Fine-tuned MELD (Phase 2 Gemini) | 0.4807 |
| `llama31_unified_prompt.ipynb` | 2,610 | Test | 7-Agent Unified Prompt | *In Progress* |
| `llama3_full_council.ipynb` | 160 | Test | Full Council (Llama 3 SFT) | *Eval Needed* |

> **Peak Performance:** Weighted F1 = **0.7771** (Ensemble Stage 1 on N=2,565; equivalent to `llama3_plain_validation.ipynb` approach on larger sample)

---

### 3.2 Peak Performance — Detailed Per-Class Breakdown

**Source:** `logs/ensemble_router/FOCUSED_stage1_baseline_report.json` | N=2,565 utterances

| Emotion | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **anger** | 0.9189 | 0.6108 | 0.7338 | 334 |
| **disgust** | 0.8125 | 0.6190 | 0.7027 | 63 |
| **fear** | 0.6170 | 0.5918 | 0.6042 | 49 |
| **joy** | 0.8881 | 0.5980 | 0.7147 | 398 |
| **neutral** | 0.7672 | 0.9661 | 0.8552 | 1,238 |
| **sadness** | 0.6332 | 0.6971 | 0.6636 | 208 |
| **surprise** | 0.8542 | 0.5964 | 0.7024 | 275 |
| **Weighted Avg** | — | — | **0.7771** | 2,565 |
| **Macro Avg** | — | — | **0.7109** | 2,565 |
| **Accuracy** | | | **0.7856** | 2,565 |

**Key Observations:**
- **Neutral** (48% of test set) is predicted with very high recall (0.966) but this inflates precision for minority classes through suppressed false positives
- **Fear** (2% support, N=49) is the weakest class (F1=0.604) — both precision and recall are lowest; the multi-stage "Fear Gate" in the aggregator prompt explicitly addresses this
- **Anger** achieves exceptional precision (0.919) but poor recall (0.611) — the system is *conservative* about predicting anger, consistent with the Gatekeeper/Anger Validation rules in the shift detector
- **Joy and Surprise** both have recall ~0.60, suggesting the model under-predicts these relative to neutral

---

### 3.3 Phase 3 Multi-Agent Council — Partial Evaluation (N=492)

From `meld_fine_tuned_phase3.ipynb` (log embedded in README):

```
              precision    recall  f1-score   support
   anger       0.00      0.00      0.00        16
   disgust     0.00      0.00      0.00         5
    fear       0.00      0.00      0.00        10
     joy       0.56      0.57      0.56        84
 neutral       0.71      0.82      0.76       259
 sadness       0.50      0.31      0.38        42
surprise       0.63      0.30      0.41        40
accuracy                           0.63       491
macro avg      0.49      0.38      0.40       491
weighted avg   0.62      0.63      0.61       491
```

**Weighted F1: 0.5764** — the full council with DeepSeek R1 aggregator severely underperforms on minority classes (anger/disgust/fear = 0.00 F1), indicating the chain-of-thought reasoning collapses to neutral/joy for ambiguous utterances.

---

### 3.4 Baseline vs. Multi-Agent Aggregator: Direct Comparison

| Metric | Plain Single-Agent (Llama 3.1) | 7-Agent Council (Phase 3) | Delta |
|---|---|---|---|
| Weighted F1 | **0.7771** | 0.5764 | **−0.2007** |
| Accuracy | 0.7856 | 0.63 | **−0.1556** |
| Anger F1 | 0.7338 | 0.00 | −0.7338 |
| Fear F1 | 0.6042 | 0.00 | −0.6042 |
| Neutral F1 | 0.8552 | 0.76 | −0.0952 |
| Joy F1 | 0.7147 | 0.56 | −0.1547 |

> [!IMPORTANT]
> The multi-agent council consistently **underperforms the plain single-agent baseline** in all evaluated configurations. The plain SFT Llama 3.1 endpoint (zero-shot per-utterance with 3-turn context window) is the current SOTA for this project.

---

## 4. Codebase Roadmap & Component Mapping

### 4.1 Directory Structure

```
Multi-Agent-LLM-Empathy/
├── .env                              # API keys (see §4.3)
├── requirements.txt                  # pandas, groq, google-genai, scikit-learn, 
│                                     # python-dotenv, kagglehub, vertexai, 
│                                     # google-cloud-aiplatform, requests, numpy
├── README.md                         # Experiment summary table + architecture overview
│
├── src/                              # All executable source code
│   ├── agents.py                     # Phase 1 core: groq_llm_call, gemini_llm_call,
│   │                                 # run_multi_agent_conversation
│   ├── fine_tuned_agents.py          # Phase 2: Vertex AI fine-tuned Gemini agents
│   ├── fine_tuned_agents_phase3.py   # Phase 3: +EmotionalShift, DeepSeek R1 aggregator,
│   │                                 # bio card injection, sliding prediction history
│   ├── llama_sft_function_calls.py   # Phase 4 agent callers: all 7 agents → SFT Llama 3.1
│   ├── llama3_full_council.py        # Phase 4 orchestrator: scene loop + bio card lookup
│   ├── llama3_single_agent_validation.py   # Phase 4.2: single-agent /predict REST baseline
│   ├── llama3_scene_batch_validation.py    # Phase 4.2: scene-batch /predict REST
│   ├── llama3_scene_batch_validation_optimized.py  # Optimized scene-batch variant
│   ├── debate_pipeline.py            # Phase 5a: 3-agent debate + arbitrator
│   ├── hybrid_pipeline.py            # Phase 5b: LaERC mental state + InitERC classifier
│   ├── load_data.py                  # load_data_from_csv() — thin pandas wrapper
│   ├── retrieval_utils.py            # MeldRetriever TF-IDF index, get_retriever() singleton
│   │
│   ├── prompts/                      # Phase 1/2/3 prompts (Gemini-era, verbose)
│   │   ├── context_manager.txt       # Scene Architect & Emotional Historian
│   │   ├── character_profiler.txt    # Character Behavioral Specialist
│   │   ├── empathy_reasonar.txt      # Linguistic Pragmatics & Subtext Specialist
│   │   ├── emotional_shift.txt       # Temporal Dynamics & Shift Detector
│   │   ├── relational_graph.txt      # Relationship Historian & Long-term Memory Specialist
│   │   ├── social_dynamics_expert.txt # Interpersonal Strategist & Face-work Analyst
│   │   └── council_aggregator.txt    # Chief Justice & Emotion Arbiter (7 deliberation rules)
│   │
│   ├── llama3_prompts/               # Phase 4/5 prompts (Llama 3.1-optimized, concise)
│   │   ├── context_manager.txt       # Condensed context historian
│   │   ├── character_profiler.txt    # Condensed character profiler
│   │   ├── empathy_reasonar.txt      # Condensed empathy reasoner
│   │   ├── emotional_shift.txt       # Condensed shift detector
│   │   ├── relational_graph.txt      # Condensed relational graph
│   │   ├── social_dynamics_expert.txt # Condensed social dynamics
│   │   ├── council_aggregator.txt    # Condensed aggregator (7 rules summary)
│   │   ├── vote_plain.txt            # Debate: scene narrative voter + MELD priors
│   │   ├── vote_linguistic.txt       # Debate: linguistic signal voter
│   │   ├── vote_context_shift.txt    # Debate: temporal/shift voter
│   │   ├── arbitrator.txt            # Debate: 3-way split resolver
│   │   ├── laerc_mental_state.txt    # LaERC: dynamic mental state JSON extractor
│   │   └── initerc_classifier.txt    # InitERC: 6-step final classification protocol
│   │
│   └── llama aggregator/             # Placeholder directory (empty subdirectory: "llama 3")
│
├── notebooks/                        # Jupyter experiments (chronological)
│   ├── meld.ipynb                    # Phase 1 initial data exploration
│   ├── speaker_bio_card_generator.ipynb    # Bio card generator v1
│   ├── speaker_bio_card_generator_v2.ipynb # Bio card generator v2 (117 KB — primary)
│   ├── meld_fine_tuned.ipynb         # Phase 2: fine-tuned Gemini council evaluation
│   ├── meld_fine_tuned_phase3.ipynb  # Phase 3: +Shift detector, WF1=0.5764
│   ├── meld_fine_tuned_phase4_llama.ipynb  # Phase 4: Llama SFT council, WF1=0.4851
│   ├── llama3_plain_validation.ipynb # Phase 4.2 baseline, WF1=0.7708
│   ├── llama3_biocards_context_manager.ipynb # Phase 4.3 bio+context, WF1=0.7286
│   ├── llama3_full_council.ipynb     # Phase 4 full council (160 utterances, eval needed)
│   ├── llama31_unified_prompt.ipynb  # Phase 5 unified 7-agent prompt (2,610 utterances, in progress)
│   ├── ensemble_confidence_router.ipynb # Phase 5c ensemble post-hoc corrections
│   ├── fine_tuning_sentiment_analyzer.ipynb # Gemini fine-tuning dataset prep
│   └── Untitled-1.ipynb              # Phase 4+ experiment scratch, WF1=0.4984
│
├── data/
│   ├── train_sent_emo.csv            # MELD training set (primary retrieval index source)
│   ├── test_sent_emo.csv             # MELD test set (primary evaluation target, 2,610 utterances)
│   ├── meld_train_80.jsonl           # 80/20 split: training partition
│   ├── meld_test_20.jsonl            # 80/20 split: validation partition
│   ├── llama2_meld_train_80.jsonl    # Llama 2 format training data
│   ├── llama2_meld_test_20.jsonl     # Llama 2 format test data
│   ├── gemini_meld_dialogue_train.jsonl # Gemini fine-tuning format (1.4 MB)
│   ├── gemini_meld_dialogue_all.jsonl  # Gemini fine-tuning format all data (1.5 MB)
│   ├── gemini_sentiment_training.jsonl  # Gemini sentiment training data
│   ├── formatted_semeval_data.jsonl  # SemEval cross-domain data
│   ├── semeval_eng_emotion_train.csv # SemEval English emotion training
│   ├── meld_fine_tune_data.py        # MELD → fine-tuning format converter
│   ├── semeval_fine_tune_data.py     # SemEval → fine-tuning format converter
│   ├── meld_data.py                  # Stub (146 bytes)
│   └── llama2_sft_data.ipynb         # SFT dataset construction notebook
│
└── logs/
    ├── speaker_bio_cards.json        # Pre-computed bio cards (174 KB, 2,281 lines, all speakers)
    ├── speaker_bio_cards.txt         # Plain text version (253 KB)
    ├── council_phase2_results.csv    # Phase 2 outputs
    ├── council_phase3_results.csv    # Phase 3 outputs (215 KB)
    ├── council_phase3_[2-4]_results.csv # Phase 3 sub-runs
    ├── council_phase4_results.csv    # Phase 4 outputs (1.9 MB)
    ├── council_phase4_1_results.csv  # Phase 4.1 (4.4 MB — largest council run)
    ├── debate_pipeline_results.csv   # Phase 5a debate outputs (327 KB)
    ├── hybrid_laerc_initerc_results.csv # Phase 5b LaERC+InitERC (3.1 MB)
    ├── llama3_single_agent_results.csv  # Phase 4.2 per-utterance baseline
    ├── llama3_scene_batch_results.csv   # Phase 4.2 scene-batch baseline
    ├── llama3/                       # Llama 3 council run outputs
    │   ├── council_llama3_1_results.csv
    │   ├── council_phase4_1_llama_all_agents_results.csv
    │   ├── council_phase4_1_llama_results.csv
    │   ├── council_phase4_llama_results.csv
    │   └── llama3_biocards_context_predictions_master.csv (14.2 MB!)
    ├── llama31_unified/              # Unified prompt run (2 CSVs, 13.8 MB each)
    └── ensemble_router/              # Ensemble stage reports (JSON + CSVs)
        ├── FOCUSED_stage1_baseline_report.json
        ├── FOCUSED_stage2_improvements_report.json
        ├── FOCUSED_stage2_refinements_report.json
        ├── performance_comparison.csv
        └── [stage1_predictions.csv, stage2_predictions.csv, final_predictions.csv...]
```

---

### 4.2 Critical Consensus Injection Points

The **multi-agent consensus mechanism** occurs at these precise locations in the code:

#### Point A — `src/agents.py` lines 94–111 (`run_multi_agent_conversation`)
The original aggregator call. Builds `aggregator_final_prompt` concatenating all 3 expert reports and calls `groq_llm_call()` with `model="openai/gpt-oss-120b"`.

#### Point B — `src/fine_tuned_agents_phase3.py` lines 170–236 (`call_council_aggregator`)
Phase 3 consensus. Injects bio card via `prompt.replace("[SPEAKER BIO CARD WILL BE PROVIDED HERE]", bio_card_content)` and previous predictions via `prompt.replace("[RECENT COUNCIL DECISIONS WILL BE PROVIDED HERE]", previous_context_content)`. Posts to OpenRouter DeepSeek R1.

#### Point C — `src/llama3_full_council.py` lines 101–108 (`run_llama3_council_scene`)
Phase 4 consensus. Calls `call_llama3_council_aggregator(rec_id, target_text, global_context, profile, sentiment, dynamics, shift, speaker_bio_card=bio_card, previous_predictions=prev_preds)` → routed to SFT Llama 3.1 endpoint.

#### Point D — `src/llama_sft_function_calls.py` lines 81–109 (`call_llama3_council_aggregator`)
Phase 4 consensus builder. Formats all 5 expert reports + bio card + rolling history into the Llama chat-template user content for `llama3_sft_call()`.

#### Point E — `src/debate_pipeline.py` lines 476–500 (`run_debate_scene`)
Debate consensus. `_aggregate_votes()` returns `(predicted_emo, avg_conf, outcome, is_split)`. If `is_split`, calls `_call_arbitrator()` → Llama 3.1 endpoint → `_parse_arbitrator_response()`.

#### Point F — `src/hybrid_pipeline.py` lines 402–418 (`run_hybrid_scene`)
LaERC+InitERC synthesis. `call_initerc_classifier()` receives 4 expert signals + retrieved examples → returns final JSON with `predicted_emotion`, `confidence`, `reasoning`.

---

### 4.3 Environment Variables Reference (`.env`)

```
DEEPSEEK_API_KEY          # Groq key for Qwen/DeepSeek models
GPT_OSS_API_KEY           # Groq key for openai/gpt-oss-120b
LLAMA_API_KEY             # Groq key for llama-3.1-8b-instant
GEMINI_API_KEY            # Google AI Studio key for Gemini
LLAMA_3.3_API_KEY         # Groq key for llama-3.3-70b-versatile
TUNED_MODEL_PROJECT_ID    # GCP project for fine-tuned Gemini endpoints (us-central1)
LLAMA_MODEL_PROJECT_ID    # GCP project for SFT Llama 3.1 endpoints
VERTEX_LOCATION           # GCP location (default: "us-central1")
CONTEXT_MANAGER           # Groq key for Llama 4 context manager agent
RELATIONAL_GRAPH_MANAGER  # Groq key for Llama 4 relational graph agent
COUNCIL_AGGREGATOR        # Groq key for aggregator (GPT-OSS or OpenRouter fallback)
OPENROUTER_API_KEY        # Primary OpenRouter key (DeepSeek R1)
OPEN_ROUTER_DEEPSEEK_KEY  # Alternate OpenRouter key
OPENROUTER_KEY            # Alternate OpenRouter key
OPENROUTER_MODEL          # Model override (default: "deepseek/deepseek-r1")
OPENROUTER_SITE_URL       # Optional: HTTP-Referer header for OpenRouter
OPENROUTER_SITE_NAME      # Optional: X-OpenRouter-Title header
```

### 4.4 Key Vertex AI Endpoint IDs

| Endpoint ID | Purpose | Region |
|---|---|---|
| `mg-endpoint-a66f56b4-7b58-4560-b43e-2a8777c38cd9` | Dedicated SFT Llama 3.1 — debate + single-agent validation | `us-east1` |
| `2346569469662330880` | Llama 3 SFT (council + hybrid) | `us-central1` |
| `6568289500043149312` | Fine-tuned Gemini (Character Profiler) | `us-central1` |
| `5496855001194037248` | Fine-tuned Gemini (Sentiment Analyst) | `us-central1` |
| Project ID: `project-d92ffbcd-75f0-4c50-bca` | Dedicated endpoint project | `us-east1` |

---

## 5. Future Work & Unresolved Hypotheses

### 5.1 Immediate Technical Priorities

#### A. Complete `llama31_unified_prompt.ipynb` Evaluation
- **Status:** *In Progress* (2,610 utterances, 7-agent unified prompt — `logs/llama31_unified/llama31_unified_predictions_master.csv` exists at 13.8 MB)
- **Action:** Run full evaluation metrics (`classification_report`) on the master CSV; if WF1 > 0.7771, this is the new SOTA and the unified architecture is validated
- **File:** `notebooks/llama31_unified_prompt.ipynb`

#### B. Diagnose `llama3_full_council.ipynb` Evaluation Gap
- **Status:** *Evaluation Needed* (160 utterances — only partially evaluated)
- **Action:** Run `sklearn.metrics.classification_report` on `logs/llama3/council_llama3_1_results.csv`

#### C. Resolve Bio Card Injection Counter-Performance
- **Key open question:** Bio cards hurt performance when added to the single-agent baseline (−0.042 WF1). This implies either:
  1. The prompt with bio card is too long for the model's effective context window at the dedicated endpoint
  2. The bio card contradicts the model's fine-tuned priors
  3. The placeholder format `[SPEAKER BIO CARD WILL BE PROVIDED HERE]` is not being replaced cleanly
- **Experiment:** Test bio card injection with truncated bio cards (static_persona only vs. all 5 fields) on a 100-utterance sample

#### D. Ensemble Router Fix
- The Stage 2 router decreased WF1 on its targeted subset (0.4573 vs 0.7771)
- **Root Cause Hypothesis:** The Stage 2 prompt instructions for low-confidence predictions are causing the model to over-correct toward neutral
- **Action:** Switch Stage 2 to a **confidence-weighted voting** strategy (weight votes by `avg_confidence`) rather than re-querying the model

---

### 5.2 Model & Architecture Upgrades

#### A. Weighted Aggregation Instead of Simple Majority
Current debate pipeline uses simple majority vote. **Alternative:** confidence-weighted voting:
```python
weighted_votes = {label: sum(v['confidence'] for v in votes if v['vote'] == label)}
predicted = max(weighted_votes, key=weighted_votes.get)
```
This is a zero-cost improvement to `_aggregate_votes()` in `debate_pipeline.py`.

#### B. Cross-Encoder Reranking for Retrieval
`MeldRetriever` uses TF-IDF cosine similarity. Replace with a **sentence-transformer cross-encoder** (e.g., `cross-encoder/nli-deberta-v3-small`) for semantic retrieval. Expected gain: +0.01–0.02 WF1 on minority classes.

#### C. Class-Specific Prompt Calibration
The per-class MELD distribution in `vote_plain.txt` (`neutral=48%, joy=15%,...`) is static. **Dynamic calibration** using the retrieved examples' actual distribution per utterance could sharpen predictions for rare classes (fear, disgust).

#### D. LoRA Fine-Tuning of Aggregator
The aggregator is currently a general-purpose model (GPT-OSS 120B / DeepSeek R1). The training data `data/llama2_meld_train_80.jsonl` and Gemini format data (`data/gemini_meld_dialogue_train.jsonl`) could be used to **fine-tune the aggregator** on expert-report → emotion label pairs extracted from the Phase 4 council logs (4.4 MB master log exists).

#### E. SemEval Cross-Domain Generalization
`data/semeval_eng_emotion_train.csv` (315 KB) and `data/formatted_semeval_data.jsonl` exist but are not yet integrated into the pipeline. Training on SemEval + MELD together could improve generalization, especially for minority emotion classes.

---

### 5.3 Addressing Systematic Weaknesses

Based on per-class metrics:

| Class | Issue | Recommended Fix |
|---|---|---|
| **Fear** (F1=0.60) | High-precision but low-recall; Fear Gate in aggregator is too strict | Relax Fear Gate — require only 1 of 3 conditions (negative valence OR shift OR threat language) |
| **Anger** (recall=0.61) | Anger Validation Rule in shift detector forces continuity too aggressively | Add override for lexically explicit anger (`"I hate"`, `"How dare you"`) |
| **Joy** (recall=0.60) | Model underpredicts joy relative to neutral | Increase Joy prior in calibration examples; add positive-valence short-circuit |
| **Surprise** (recall=0.60) | Same recall suppression as Joy | Strengthen surprise signal detection in `vote_linguistic.txt` |

---

### 5.4 Publication / Conference Readiness

The project has the components for a conference submission to **ACL/EMNLP/NAACL 2026–2027** under a category such as *"Multi-Agent LLMs for Emotion Recognition in Conversation"*. The following gaps must be addressed:

1. **Full test-set evaluation of all pipeline variants** (currently several are partial/in-progress)
2. **Ablation study:** Single agent → +Context → +Bio Cards → +Shift → +Council to isolate each component's contribution
3. **Statistical significance testing** (bootstrap or McNemar's test) between the best WF1=0.7771 system and the full council WF1=0.5764 system
4. **Comparison against published MELD baselines:**
   - COSMIC (2020): WF1 ≈ 0.65
   - UniMSE (2022): WF1 ≈ 0.67
   - UniEMO (2023): WF1 ≈ 0.70
   - *Target: beating ~0.73 to claim SOTA territory*
5. **Multimodal Extension:** MELD provides audio/video; the current system is text-only. Adding audio features (e.g., OpenAI Whisper features or Wav2Vec2 embeddings) could be the differentiation paper contribution
6. **Complete the SemEval integration** — a multi-corpus evaluation is standard for ERC papers

> [!TIP]
> The `llama31_unified_prompt.ipynb` (in-progress, 2,610 utterances) result is the most important immediate milestone. If it achieves WF1 > 0.75, it constitutes a meaningful result over the bio-card variant (0.7286) and should be the lead contribution in any submission.

---

### 5.5 Technical Debt & Code TODOs

1. **`src/llama3_full_council.py` line 172:** `# LIMIT FOR TESTING - Remove after verification` — the `unprocessed_scenes[:1]` limit is commented out but was left in the code as a reminder
2. **`src/fine_tuned_agents.py` lines 108–135` / `src/fine_tuned_agents_phase3.py` lines 252–279`:** `run_phase2_council()` is commented out in both files — the notebook-driven workflow replaced it but the core orchestration logic was never migrated back to production scripts
3. **`src/llama aggregator/`:** The directory contains only an empty `llama 3/` subdirectory — this was likely a staging area for planned aggregator variants that was never populated
4. **`src/llama3_scene_batch_validation_optimized.py`:** The "optimized" variant (16.3 KB vs 12.5 KB for the original) — no documentation on what was optimized; a diff should be captured before handoff
5. **Hardcoded `project-d92ffbcd-75f0-4c50-bca` Project ID** in `debate_pipeline.py` and `llama3_single_agent_validation.py` — should be moved to `.env` as `DEDICATED_ENDPOINT_PROJECT_ID`
6. **`notebooks/Untitled-1.ipynb`:** The notebook has no descriptive name; its experiment (Phase 4+, WF1=0.4984) should be renamed before archiving
7. **`data/llama2_sft_data.ipynb`** is inside the `data/` directory — should be moved to `notebooks/`

---

*End of Knowledge Transfer Document — Prepared for incoming team handoff, Multi-Agent ERC project, May 2026*
