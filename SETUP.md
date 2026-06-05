# Setup Guide

This guide walks you through getting the project running from scratch, including Python environment setup, API key configuration, GCP authentication for Vertex AI, and instructions for running each pipeline.

---

## Prerequisites

| Requirement | Version / Notes |
|-------------|----------------|
| Python | 3.10 or higher |
| Google Cloud SDK (`gcloud`) | Latest — [install here](https://cloud.google.com/sdk/docs/install) |
| GCP Project | With Vertex AI API enabled |
| MELD dataset | See [Getting the data](#2-getting-the-data) below |

---

## 1. Clone and create a virtual environment

```bash
git clone https://github.com/deepanshBatra/Multi-Agent-LLM-Empathy.git
cd Multi-Agent-LLM-Empathy

# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` installs:

| Package | What it's used for |
|---------|-------------------|
| `pandas` | Data loading and result CSVs |
| `numpy` | TF-IDF similarity calculations |
| `scikit-learn` | TF-IDF retriever + F1 evaluation |
| `groq` | Calling Groq-hosted models (Llama, GPT-OSS) |
| `google-genai` | Gemini API calls |
| `google-cloud-aiplatform` | Vertex AI model endpoints |
| `vertexai` | Vertex AI SDK (Llama SFT endpoints) |
| `requests` | Direct HTTP calls to Vertex AI predict endpoints |
| `python-dotenv` | Loading API keys from `.env` |
| `kagglehub[pandas-datasets]` | Downloading MELD from Kaggle (optional) |

---

## 2. Getting the data

The `data/` directory is gitignored. You need to obtain the MELD files separately and place them here:

```
data/
├── train_sent_emo.csv      # MELD training set (~11,000 utterances)
├── test_sent_emo.csv       # MELD test set (2,610 utterances)
├── meld_train_80.jsonl     # Llama SFT training split (80%)
└── meld_test_20.jsonl      # Llama SFT eval split (20%)
```

### Option A — Download from Kaggle

```python
import kagglehub
path = kagglehub.dataset_download("zaber666/meld-dataset")
```

Then copy `train_sent_emo.csv` and `test_sent_emo.csv` into `data/`.

### Option B — Direct download

MELD is publicly available from the [original paper repository](https://github.com/SenticNet/MELD). Download and place the CSV files into `data/`.

> The `.jsonl` files (`meld_train_80.jsonl`, `meld_test_20.jsonl`) were generated from the raw MELD CSVs during fine-tuning. See `data/meld_fine_tune_data.py` if you need to regenerate them.

---

## 3. API keys

Create a `.env` file in the project root. This file is gitignored and will never be committed.

```bash
# .env

# ── Groq API (for Llama 3.x, GPT-OSS, Qwen via Groq) ──────────────────────
LLAMA_API_KEY=gsk_...          # Groq key for Llama 3.1 8B
LLAMA_3.3_API_KEY=gsk_...      # Groq key for Llama 3.3 70B
CONTEXT_MANAGER=gsk_...        # Groq key for context manager agent
RELATIONAL_GRAPH_MANAGER=gsk_... # Groq key for relational graph agent
COUNCIL_AGGREGATOR=gsk_...     # Groq key for GPT-OSS aggregator

# ── Gemini (Google AI Studio) ───────────────────────────────────────────────
GEMINI_API_KEY=AIza...

# ── OpenRouter (DeepSeek R1, Phase 3 aggregator) ───────────────────────────
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=deepseek/deepseek-r1

# ── DeepSeek direct (Phase 1 Empathy Reasoner) ─────────────────────────────
DEEPSEEK_API_KEY=sk-...

# ── Vertex AI (fine-tuned Llama + Gemini endpoints) ────────────────────────
TUNED_MODEL_PROJECT_ID=your-gcp-project-id
LLAMA_MODEL_PROJECT_ID=your-gcp-project-id
VERTEX_LOCATION=us-central1
```

### Which keys you actually need

You don't need every key to run every experiment. Here's what each pipeline requires:

| Pipeline / Notebook | Keys needed |
|---------------------|------------|
| `meld.ipynb` (Phase 1) | `LLAMA_3.3_API_KEY`, `LLAMA_API_KEY`, `DEEPSEEK_API_KEY`, `COUNCIL_AGGREGATOR` |
| `meld_fine_tuned.ipynb` (Phase 2) | `TUNED_MODEL_PROJECT_ID` + GCP auth, `COUNCIL_AGGREGATOR` |
| `meld_fine_tuned_phase3.ipynb` (Phase 3) | Above + `OPENROUTER_API_KEY` |
| `llama3_plain_validation.ipynb` | `LLAMA_MODEL_PROJECT_ID` + GCP auth |
| `llama3_biocards_context_manager.ipynb` | `LLAMA_MODEL_PROJECT_ID` + GCP auth |
| `debate_pipeline.py` | `LLAMA_MODEL_PROJECT_ID` + GCP auth |
| `hybrid_pipeline.py` | `LLAMA_MODEL_PROJECT_ID` + GCP auth |

> **Groq** provides free-tier access for most of its hosted models. Sign up at [console.groq.com](https://console.groq.com).  
> **OpenRouter** routes to DeepSeek R1 and other models. Sign up at [openrouter.ai](https://openrouter.ai).

---

## 4. Google Cloud / Vertex AI setup

Several pipelines call fine-tuned models deployed on Vertex AI. You need to authenticate locally so the code can call those endpoints.

### Step 1 — Install and initialise gcloud

```bash
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
```

### Step 2 — Authenticate Application Default Credentials (ADC)

The Python scripts use ADC for all Vertex AI calls:

```bash
gcloud auth application-default login
```

This writes credentials to a local file that `google.auth.default()` picks up automatically. You only need to do this once per machine (tokens auto-refresh).

### Step 3 — Enable APIs

In your GCP project, make sure these APIs are enabled:

```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable logging.googleapis.com
```

### Step 4 — Verify the setup

```bash
gcloud ai models list --region=us-central1
```

If this returns a list (even empty), your auth is working.

---

## 5. Running the experiments

### Notebooks (recommended starting point)

Launch Jupyter and open any notebook in `notebooks/`:

```bash
jupyter notebook
```

The recommended order for understanding the project progression:

1. `meld.ipynb` — simplest multi-agent setup, understand the architecture
2. `llama3_plain_validation.ipynb` — best result (0.7708 WF1), the baseline to beat
3. `meld_fine_tuned_phase3.ipynb` — fine-tuned council with shift detection
4. `ensemble_confidence_router.ipynb` — 2-stage confidence routing

---

### Python scripts

All scripts are run as modules from the project root (so relative imports in `src/` resolve correctly). Always activate your virtual environment first.

#### Llama 3.1 — per-utterance validation

Calls the fine-tuned Llama 3.1 endpoint once per utterance, with 3-turn conversational context.

```bash
.venv\Scripts\python.exe -m src.llama3_single_agent_validation
# Results → logs/llama3_single_agent_results.csv
```

#### Llama 3.1 — scene-batch validation

Sends an entire scene to the model in one call, asking it to predict all utterances at once.

```bash
.venv\Scripts\python.exe -m src.llama3_scene_batch_validation
# Results → logs/llama3_scene_batch_results.csv
```

#### Scene-batch + Bio Cards + TF-IDF (optimized)

The same scene-batch pipeline enriched with character bio cards and dynamic few-shot retrieval from the training set.

```bash
.venv\Scripts\python.exe -m src.llama3_scene_batch_validation_optimized
# Results → logs/llama3_scene_batch_results_optimized.csv
```

#### Debate Pipeline (3-agent vote + arbitrator)

Three agents independently vote on each utterance. Majority wins. 3-way splits go to an arbitrator.

```bash
.venv\Scripts\python.exe -m src.debate_pipeline
# Results → logs/debate_pipeline_results.csv
```

#### Hybrid LaERC + InitERC Pipeline

4-stage pipeline: TF-IDF retrieval → Empathy Reasoner + Shift Detector (parallel) → LaERC Mental State → InitERC Classifier.

```bash
.venv\Scripts\python.exe -m src.hybrid_pipeline
# Results → logs/hybrid_laerc_initerc_results.csv
```

#### Full Llama 3.1 Council

7-agent council where every agent is the same fine-tuned Llama 3.1 model.

```bash
.venv\Scripts\python.exe -m src.llama3_full_council
# Results → logs/council_llama3_1_results.csv
```

---

### Resume support

All script-based pipelines write results incrementally after each scene and can resume from where they left off if interrupted. Just re-run the same command — the script reads any existing output file and skips scenes already processed.

---

## 6. Evaluating results

Every pipeline prints a classification report and Weighted F1 score at the end of its run. To re-evaluate an existing results CSV manually:

```python
import pandas as pd
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv("logs/debate_pipeline_results.csv")
df = df.drop_duplicates(subset=["Recognition_ID"], keep="last")

y_true = df["Actual_Emotion"].str.lower().str.strip()
y_pred = df["predicted_emotion"].str.lower().str.strip()

wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
print(f"Weighted F1: {wf1:.4f}")
print(classification_report(y_true, y_pred, zero_division=0))
```

> Column names vary slightly between pipelines — use `df.columns` to check if the prediction column is `predicted_emotion` or `Predicted_Emotion`.

---

## 7. Regenerating speaker bio cards

The bio cards in `logs/speaker_bio_cards.json` are pre-generated and committed to the repo. If you want to regenerate them (e.g., to add new characters), use:

```bash
jupyter notebook notebooks/speaker_bio_card_generator_v2.ipynb
```

This runs the fine-tuned Llama 3.1 model over the unique speakers in the training set and writes structured JSON profiles for each character.

---

## 8. Project-wide environment variables reference

| Variable | Used by | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | Phase 1, agents.py | Google AI Studio API key |
| `DEEPSEEK_API_KEY` | Phase 1 | DeepSeek / OpenRouter key for Qwen |
| `LLAMA_API_KEY` | Phase 1 | Groq key for Llama 3.1 8B |
| `LLAMA_3.3_API_KEY` | Phase 1 | Groq key for Llama 3.3 70B |
| `GPT_OSS_API_KEY` | Phase 1 | Groq key for GPT-OSS 120B |
| `CONTEXT_MANAGER` | Phase 2 | Groq key for Context Manager agent |
| `RELATIONAL_GRAPH_MANAGER` | Phase 2 | Groq key for Relational Graph agent |
| `COUNCIL_AGGREGATOR` | Phase 2 | Groq key for Council Aggregator / also doubles as OpenRouter key |
| `OPENROUTER_API_KEY` | Phase 3 | OpenRouter key (DeepSeek R1) |
| `OPENROUTER_MODEL` | Phase 3 | Model string, e.g. `deepseek/deepseek-r1` |
| `TUNED_MODEL_PROJECT_ID` | Phase 2, 3 | GCP project with fine-tuned Gemini endpoints |
| `LLAMA_MODEL_PROJECT_ID` | Phases 4+ | GCP project with fine-tuned Llama 3.1 endpoints |
| `VERTEX_LOCATION` | Hybrid pipeline | Vertex AI region, default `us-central1` |

---

## 9. Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**  
Run scripts from the project root as modules, not as direct files:
```bash
# ✅ Correct
.venv\Scripts\python.exe -m src.debate_pipeline

# ❌ Wrong
.venv\Scripts\python.exe src/debate_pipeline.py
```

**`google.auth.exceptions.DefaultCredentialsError`**  
Run `gcloud auth application-default login` again. ADC credentials may have expired.

**`429 / quota exceeded` from Vertex AI**  
The pipelines have built-in exponential backoff for 429 errors. If you're consistently hitting quota, reduce `MAX_SCENE_WORKERS` in the relevant script (default is 3).

**`KeyError` when reading results CSV**  
Different pipelines use slightly different column names (`predicted_emotion` vs `Predicted_Emotion`). Check `df.columns` and adjust your evaluation code accordingly.

**Predictions all coming back as `neutral`**  
This usually means the endpoint is returning malformed JSON. Check the `raw_output` column in the results CSV to see what the model actually returned, and verify the endpoint is healthy.

**`data/` files not found**  
The `data/` directory is gitignored. See [Getting the data](#2-getting-the-data) above.
