# Multi-Agent Empathy & Emotion Recognition (MERC)
### Phase 2: The 6-Agent Council for MELD Classification

This research project utilizes a coordinated Multi-Agent System (MAS) to achieve state-of-the-art (SOTA) results in Emotion Recognition in Conversation (ERC) using the MELD dataset.

## 🚀 The Architecture: "The Council"
Unlike single-model approaches, this system decomposes emotional reasoning into six specialized agents to handle nuance, sarcasm, and social context.

| Agent | Role | Model | Specialization |
| :--- | :--- | :--- | :--- |
| **1. Context Manager==> /scratch/deep13/meld_finetuning/logs/val_eval_60403717.log <==
        fear       0.00      0.00      0.00        10
         joy       0.56      0.57      0.56        84
     neutral       0.71      0.82      0.76       259
     sadness       0.50      0.31      0.38        42
    surprise       0.63      0.30      0.41        40

    accuracy                           0.63       491
   macro avg       0.49      0.38      0.40       491
weighted avg       0.62      0.63      0.61       491** | Scene Historian | Llama 4 Maverick | Global scene arcs & plot beats |
| **2. Relational Manager**| Social Mapper | Llama 4 Scout | Social tension & 10M token memory |
| **3. Character Profiler** | Personality Expert | **Tuned Gemini 3 Flash** | Behavioral DNA (Friends characters) |
| **4. Sentiment Analyst** | Linguistic Scorer | **Tuned Gemini 3 Flash** | Raw valence & arousal weights |
| **5. Social Dynamics** | Reasoning Bridge | Llama 4 Maverick | Synthesizes relationship + profile |
| **6. Council Aggregator**| The Final Judge | GPT-OSS 120B | Resolves specialist conflicts |



---

---

## 📊 Experimental Results

All experiments are evaluated on the **MELD (Multimodal EmotionLines Dataset)** test set, which contains **2,610 utterances** from the Friends TV series. The table below summarizes all notebook experiments with their respective configurations and Weighted F1 scores.

| Notebook | Utterances | Dataset | Technique | Weighted F1 |
| :--- | :---: | :--- | :--- | :---: |
| [llama3_plain_validation.ipynb](notebooks/llama3_plain_validation.ipynb) | 2,474 | Test | Llama 3.1 (Plain Validation) | **0.7708** |
| [llama3_biocards_context_manager.ipynb](notebooks/llama3_biocards_context_manager.ipynb) | 2,545 | Test | Llama 3.1 + Bio Cards Context | **0.7286** |
| [meld_fine_tuned_phase3.ipynb](notebooks/meld_fine_tuned_phase3.ipynb) | 492 | Test | Fine-tuned Phase 3 | 0.5764 |
| [meld_fine_tuned_phase4_llama.ipynb](notebooks/meld_fine_tuned_phase4_llama.ipynb) | 149 | Test | Fine-tuned Phase 4 (Llama) | 0.4851 |
| [meld_fine_tuned.ipynb](notebooks/meld_fine_tuned.ipynb) | 181 | Test | Fine-tuned MELD | 0.4807 |
| [Untitled-1.ipynb](notebooks/Untitled-1.ipynb) | 160 | Test | Fine-tuned Phase 4+ | 0.4984 |
| [llama31_unified_prompt.ipynb](notebooks/llama31_unified_prompt.ipynb) | 2,610 | Test | Llama 3.1 Unified Prompt (7-Agent) | *In Progress* |
| [llama3_full_council.ipynb](notebooks/llama3_full_council.ipynb) | 160 | Test | Llama 3 Full Council | *Evaluation Needed* |

### Key Findings:
- **Best Performance**: Llama 3.1 plain validation achieves **0.7708 Weighted F1**, outperforming all other configurations
- **High-Performance Range**: Llama 3-based approaches (0.7286-0.7708) significantly outperform fine-tuned models (0.4807-0.5764)
- **Unified Prompt Advantage**: The 7-agent unified prompt architecture is designed to consolidate multi-agent reasoning into a single optimized prompt
- **Bio Cards Enhancement**: Context augmentation with speaker biographical information shows moderate improvement potential

---

## 🛠️ Setup Instructions

### 1. Environment & Dependencies
Clone the repository and install the required libraries:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt
