import os
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv

load_dotenv()

# Configuration from User Feedback
PROJECT_ID = os.getenv("LLAMA_MODEL_PROJECT_ID") or os.getenv("TUNED_MODEL_PROJECT_ID")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
ENDPOINT_ID = "2346569469662330880"

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION)
llama3_model = GenerativeModel(f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}")

# Prompt Loading Helper
def load_prompt(file_address):
    with open(file_address, "r") as f:
        return f.read()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(BASE_DIR, "llama3_prompts")

# Load all concise prompts
prompts = {
    "character_profiler": load_prompt(os.path.join(PROMPTS_DIR, "character_profiler.txt")),
    "context_manager": load_prompt(os.path.join(PROMPTS_DIR, "context_manager.txt")),
    "council_aggregator": load_prompt(os.path.join(PROMPTS_DIR, "council_aggregator.txt")),
    "emotional_shift": load_prompt(os.path.join(PROMPTS_DIR, "emotional_shift.txt")),
    "empathy_reasonar": load_prompt(os.path.join(PROMPTS_DIR, "empathy_reasonar.txt")),
    "relational_graph": load_prompt(os.path.join(PROMPTS_DIR, "relational_graph.txt")),
    "social_dynamics_expert": load_prompt(os.path.join(PROMPTS_DIR, "social_dynamics_expert.txt")),
}

def llama3_sft_call(system_instruction, user_content):
    """Formats the prompt using Llama 3 chat template and calls the SFT endpoint."""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    response = llama3_model.generate_content(prompt)
    return response.text

# Specialized Agent Callers
def call_llama3_profiler(utterance, social_graph):
    user_content = f"Social Context: {social_graph}\nUtterance: {utterance}"
    return llama3_sft_call(prompts["character_profiler"], user_content)

def call_llama3_sentiment(utterance):
    return llama3_sft_call(prompts["empathy_reasonar"], utterance)

def call_llama3_social_dynamics(utterance, profile, social_graph):
    user_content = f"UTTERANCE: {utterance}\nCHARACTER PROFILE: {profile}\nRELATIONAL GRAPH: {social_graph}"
    return llama3_sft_call(prompts["social_dynamics_expert"], user_content)

def call_llama3_emotional_shift(previous_utterance, previous_speaker, target_utterance, target_speaker, context_summary):
    user_content = f"""CONTEXT SUMMARY:
{context_summary}

PREVIOUS UTTERANCE:
Speaker: {previous_speaker}
Text: {previous_utterance}

TARGET UTTERANCE:
Speaker: {target_speaker}
Text: {target_utterance}"""
    return llama3_sft_call(prompts["emotional_shift"], user_content)

def call_llama3_context_manager(scene_script):
    user_content = f"Scene Dialogue:\n{scene_script}"
    return llama3_sft_call(prompts["context_manager"], user_content)

def call_llama3_relational_graph(scene_script):
    user_content = f"Scene Dialogue:\n{scene_script}"
    return llama3_sft_call(prompts["relational_graph"], user_content)

def call_llama3_council_aggregator(recognition_id, utterance, context, profile, sentiment, dynamics, emotional_shift, speaker_bio_card=None, previous_predictions=None):
    # Format speaker bio card
    bio_card_content = speaker_bio_card if speaker_bio_card else "[No speaker persona available]"
    
    # Format previous predictions
    previous_context_content = ""
    if previous_predictions:
        previous_context_content = "Last 3 Predictions:\n"
        for i, pred in enumerate(previous_predictions, 1):
            previous_context_content += f"{i}. '{pred.get('utterance', '')[:50]}...' | Emotion: {pred.get('emotion', '')} | {pred.get('shift', '')}\n"
    else:
        previous_context_content = "[No previous predictions available]"

    system_prompt = prompts["council_aggregator"]
    
    user_content = f"""
RECOGNITION_ID: {recognition_id}
TARGET UTTERANCE: {utterance}
SPEAKER BIO: {bio_card_content}
RECENT DECISIONS: {previous_context_content}

EXPERT REPORTS:
1. Context Historian: {context}
2. Character Profiler: {profile}
3. Sentiment Analyst: {sentiment}
4. Social Dynamics: {dynamics}
5. Emotional Shift Detector: {emotional_shift}
"""
    return llama3_sft_call(system_prompt, user_content)
