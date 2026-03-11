# Multi-Agent Empathy & Emotion Recognition (MERC)
### Phase 2: The 6-Agent Council for MELD Classification

This research project utilizes a coordinated Multi-Agent System (MAS) to achieve state-of-the-art (SOTA) results in Emotion Recognition in Conversation (ERC) using the MELD dataset.

## 🚀 The Architecture: "The Council"
Unlike single-model approaches, this system decomposes emotional reasoning into six specialized agents to handle nuance, sarcasm, and social context.

| Agent | Role | Model | Specialization |
| :--- | :--- | :--- | :--- |
| **1. Context Manager** | Scene Historian | Llama 4 Maverick | Global scene arcs & plot beats |
| **2. Relational Manager**| Social Mapper | Llama 4 Scout | Social tension & 10M token memory |
| **3. Character Profiler** | Personality Expert | **Tuned Gemini 3 Flash** | Behavioral DNA (Friends characters) |
| **4. Sentiment Analyst** | Linguistic Scorer | **Tuned Gemini 3 Flash** | Raw valence & arousal weights |
| **5. Social Dynamics** | Reasoning Bridge | Llama 4 Maverick | Synthesizes relationship + profile |
| **6. Council Aggregator**| The Final Judge | GPT-OSS 120B | Resolves specialist conflicts |



---

## 🛠️ Setup Instructions

### 1. Environment & Dependencies
Clone the repository and install the required libraries:
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt
