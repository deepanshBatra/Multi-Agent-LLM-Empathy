# Multi-Agent LLM Empathy (MERC)

> **Research project** exploring whether multi-agent LLM architectures can outperform single-model approaches at Emotion Recognition in Conversation (ERC) using the MELD (Friends TV show) dataset.

---

## What this project does

Each line of dialogue in a conversation carries emotional meaning shaped by who said it, what came before it, how the relationship between speakers works, and dozens of other subtle signals. Standard single-model approaches treat every utterance in isolation. This project asks: what happens if you decompose that reasoning across multiple specialized agents — one for character personality, one for linguistic tone, one for social dynamics — and then have them collaborate on a final verdict?

We ran 9 distinct experiments across different architectures, from simple 3-agent councils to fine-tuned specialist ensembles, voting/debate systems, and retrieval-augmented pipelines. The dataset is **MELD** (2,610 test utterances from Friends), and every experiment is evaluated using **Weighted F1** so results are directly comparable.

### Best result: **0.7708 Weighted F1** — single fine-tuned Llama 3.1 with a plain prompt

---

## Results at a glance

| Experiment | Architecture | Models | Weighted F1 |
|:-----------|:-------------|:-------|:-----------:|
| Plain Llama 3.1 (notebook) | Single agent | Fine-tuned Llama 3.1 | **0.7708** |
| Llama 3.1 + Bio Cards | Single agent + context | Fine-tuned Llama 3.1 | 0.7286 |
| Fine-Tuned Council Phase 3 | 6-agent hierarchy | Fine-tuned Gemini Flash + DeepSeek R1 | 0.5764 |
| Scene-Batch Validation | Single agent, scene-level | Fine-tuned Llama 3.1 | 0.5657 |
| Debate Pipeline | 3-agent vote + arbitrator | Fine-tuned Llama 3.1 × 4 | 0.5409 |
| Fine-Tuned Council Phase 2 | 5-agent hierarchy | Fine-tuned Gemini Flash + Groq | 0.4807 |
| Early Council (Phase 1) | 3-agent hierarchy | Llama 3.3, Qwen3, GPT-OSS | ~0.48 |
| Hybrid LaERC + InitERC | 4-stage pipeline + TF-IDF | Fine-tuned Llama 3.1 | ~0.55 |
| Unified 7-Lens Prompt | Single call, 7 reasoning lenses | Fine-tuned Llama 3.1 | *(in progress)* |

---

## Repository structure

```
Multi-Agent-LLM-Empathy/
│
├── data/                          # MELD dataset files (gitignored — obtain separately)
│   ├── train_sent_emo.csv         # Training set (~11k utterances)
│   ├── test_sent_emo.csv          # Test set (2,610 utterances, 280 scenes)
│   ├── meld_train_80.jsonl        # Llama SFT training split
│   └── meld_test_20.jsonl         # Llama SFT eval split
│
├── notebooks/                     # Jupyter experiment notebooks
│   ├── meld.ipynb                 # Exp 1: Early 3-agent Gemini council
│   ├── meld_fine_tuned.ipynb      # Exp 2a: Fine-tuned council (Phase 2)
│   ├── meld_fine_tuned_phase3.ipynb  # Exp 2b: + Emotional Shift + Bio Cards (Phase 3)
│   ├── meld_fine_tuned_phase4_llama.ipynb  # Exp 2c: Llama-powered Phase 4
│   ├── llama3_plain_validation.ipynb       # Exp 3: Best result — 0.7708 WF1
│   ├── llama3_biocards_context_manager.ipynb  # Exp 4: + Bio Cards context
│   ├── llama3_full_council.ipynb           # Exp 5: Full 7-agent council
│   ├── llama31_unified_prompt.ipynb        # Exp 6: Unified 7-lens single prompt
│   ├── ensemble_confidence_router.ipynb    # Exp 9: 2-stage confidence router
│   ├── speaker_bio_card_generator.ipynb    # Utility: generate character bio cards
│   └── speaker_bio_card_generator_v2.ipynb
│
├── src/                           # Core source code
│   ├── agents.py                  # Base LLM callers (Groq, Gemini, OpenRouter)
│   ├── load_data.py               # CSV loading utility
│   ├── retrieval_utils.py         # TF-IDF retriever (MELD training index)
│   ├── debate_pipeline.py         # Exp 7: 3-agent debate + arbitrator
│   ├── hybrid_pipeline.py         # Exp 8: LaERC + InitERC hybrid pipeline
│   ├── llama3_scene_batch_validation.py   # Llama scene-batch inference script
│   ├── llama3_scene_batch_validation_optimized.py  # + Bio Cards + TF-IDF retrieval
│   ├── llama3_single_agent_validation.py  # Per-utterance inference script
│   ├── llama3_full_council.py     # Full Llama 3.1 council orchestrator
│   ├── llama_sft_function_calls.py  # Vertex AI SFT endpoint wrappers
│   ├── fine_tuned_agents.py       # Phase 2 fine-tuned Gemini agent callers
│   ├── fine_tuned_agents_phase3.py  # Phase 3 extended agent callers
│   ├── prompts/                   # Gemini-era prompt templates
│   └── llama3_prompts/            # Llama 3.1 prompt templates
│       ├── context_manager.txt
│       ├── character_profiler.txt
│       ├── council_aggregator.txt
│       ├── emotional_shift.txt
│       ├── empathy_reasonar.txt
│       ├── relational_graph.txt
│       ├── social_dynamics_expert.txt
│       ├── vote_plain.txt
│       ├── vote_linguistic.txt
│       ├── vote_context_shift.txt
│       └── arbitrator.txt
│
├── logs/                          # Experiment output CSVs and JSON reports
│   ├── speaker_bio_cards.json     # Pre-generated character profiles
│   ├── debate_pipeline_results.csv
│   ├── hybrid_laerc_initerc_results.csv
│   ├── llama3_scene_batch_results.csv
│   ├── llama3_single_agent_results.csv
│   └── ensemble_router/           # Confidence router outputs
│
├── .env                           # API keys (gitignored)
├── requirements.txt               # Python dependencies
└── SETUP.md                       # Full setup and usage guide
```

---

## Emotion labels

All models classify into exactly 7 MELD labels:

`neutral` · `joy` · `surprise` · `anger` · `sadness` · `disgust` · `fear`

---

## Key design decisions

**Why Weighted F1?**  
MELD is heavily class-imbalanced — `neutral` makes up ~47% of utterances. Weighted F1 accounts for this by weighting each class by its support, giving a more honest picture of overall performance than accuracy.

**Why fine-tune at all?**  
We tried general instruction-following models first (Phase 1). They performed poorly (~0.48 WF1). Fine-tuning on MELD training data gave an immediate jump to ~0.77 WF1 — the most impactful single change in the whole project.

**Why did multi-agent underperform?**  
The limit seems to be the underlying model's ceiling. Once the fine-tuned Llama 3.1 reaches ~0.77 WF1 on its own, adding orchestration layers introduces compounding parsing failures and inference cost without giving the model access to genuinely new information. The main value of multi-agent is **interpretability** — you get per-utterance reasoning traces, vote breakdowns, and shift signals that a single-model output doesn't provide.

---

## Detailed experiment documentation

See [`experiment_log.md`](experiment_log.md) for a full write-up of every experiment — what we built, how each agent works, which models were used, architecture diagrams, and the results.

## Setup

See [`SETUP.md`](SETUP.md) for step-by-step environment setup, API key configuration, and instructions for running each pipeline.
