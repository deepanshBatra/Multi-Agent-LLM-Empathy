import os
import json
import concurrent.futures
import requests
import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel
from google import genai
from src.agents import * # Assuming your groq_llm_call and load_prompt are defined here

load_dotenv()

# --- 1. CONFIGURATION & AUTHENTICATION ---
# Using your verified Project ID (ends in 911 / authenticated as 693)
tuned_project_id = os.getenv("TUNED_MODEL_PROJECT_ID")
location = 'us-central1'

# Initialize Vertex AI for the fine-tuned models (Identity-based)
vertexai.init(project=tuned_project_id, location=location)

# Initialize the Unified Client for Agent 5 (Also Identity-based)
# Setting vertexai=True forces it to use your gcloud login instead of an API Key
gemini_cloud_client = genai.Client(
    vertexai=True, 
    project=tuned_project_id, 
    location=location
)

# --- 2. MODEL DEFINITIONS ---
# Agent 3: Fine-tuned Character Profiler 
profiler_model = GenerativeModel(
    f"projects/{tuned_project_id}/locations/{location}/endpoints/6568289500043149312"
)

# Agent 4: Fine-tuned Sentiment Analyzer
sentiment_model = GenerativeModel(
    f"projects/{tuned_project_id}/locations/{location}/endpoints/5496855001194037248"
)

# Other APIs
context_manager_api = os.getenv("CONTEXT_MANAGER") 
relational_graph_api = os.getenv("RELATIONAL_GRAPH_MANAGER")
openrouter_api_key = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPEN_ROUTER_DEEPSEEK_KEY")
    or os.getenv("COUNCIL_AGGREGATOR")
    or os.getenv("OPENROUTER_KEY")
)
openrouter_site_url = os.getenv("OPENROUTER_SITE_URL")
openrouter_site_name = os.getenv("OPENROUTER_SITE_NAME")
openrouter_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1")

# --- 3. PROMPT LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prompts_path = os.path.join(BASE_DIR, "prompts")

context_manager_prompt = load_prompt(os.path.join(prompts_path, "context_manager.txt"))
social_dynamics_expert_prompt = load_prompt(os.path.join(prompts_path, "social_dynamics_expert.txt"))
empathy_reasonar_prompt = load_prompt(os.path.join(prompts_path, "empathy_reasonar.txt"))
council_aggregator_prompt = load_prompt(os.path.join(prompts_path, "council_aggregator.txt"))
relational_graph_prompt = load_prompt(os.path.join(prompts_path, "relational_graph.txt"))
character_profiler_prompt = load_prompt(os.path.join(prompts_path, "character_profiler.txt"))
emotional_shift_prompt = load_prompt(os.path.join(prompts_path, "emotional_shift.txt"))

# --- 4. AGENT FUNCTIONS ---

def call_tuned_gemini(model_obj, prompt, utterance_text):
    """General caller for your fine-tuned Gemini endpoints."""
    full_prompt = f"{prompt}\n\nTARGET UTTERANCE: {utterance_text}"
    response = model_obj.generate_content(full_prompt)
    return response.text

def call_tuned_profiler(utterance, social_graph):
    context = f"Social Context: {social_graph}"
    return call_tuned_gemini(profiler_model, character_profiler_prompt, f"{context}\nUtterance: {utterance}")

def call_tuned_sentiment(utterance):
    return call_tuned_gemini(sentiment_model, empathy_reasonar_prompt, utterance)

def call_social_dynamics(utterance, profile, social_graph):
    """Agent 5: Now using the unified cloud client (No API Key needed)."""
    prompt = f"""
    {social_dynamics_expert_prompt}
    UTTERANCE: {utterance}
    CHARACTER PROFILE: {profile}
    RELATIONAL GRAPH: {social_graph}
    """
    # Uses the same identity as your fine-tuned models
    response = gemini_cloud_client.models.generate_content(
        model="gemini-2.0-flash", 
        contents=prompt
    )
    return response.text

def call_emotional_shift(previous_utterance, previous_speaker, target_utterance, target_speaker, context_summary):
    """Agent 6: Detects turn-to-turn emotional pivots."""
    prompt = f"""
    {emotional_shift_prompt}

    CONTEXT SUMMARY:
    {context_summary}

    PREVIOUS UTTERANCE:
    Speaker: {previous_speaker}
    Text: {previous_utterance}

    TARGET UTTERANCE:
    Speaker: {target_speaker}
    Text: {target_utterance}
    """
    response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {openrouter_api_key}"
  },
  data=json.dumps({
    "model": "deepseek/deepseek-r1", # Optional
    "messages": [
      {
        "role": "user",
        "content": prompt
      }
    ]
  })
)

    return response.text

def call_council_aggregator(recognition_id, utterance, context, profile, sentiment, dynamics, emotional_shift):
    """Agent 7: Synthesizes expert reports into a final MELD label via OpenRouter DeepSeek."""
    prompt = f"""
    {council_aggregator_prompt}
    
    METADATA:
    Recognition_ID: {recognition_id}
    
    TARGET UTTERANCE: {utterance}
    
    EXPERT REPORTS:
    1. Context Historian: {context}
    2. Character Profiler: {profile}
    3. Sentiment Analyst: {sentiment}
    4. Social Dynamics: {dynamics}
    5. Emotional Shift Detector: {emotional_shift}
    """

    # Re-resolve credentials at call time so notebook/session env changes are picked up.
    runtime_openrouter_api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPEN_ROUTER_DEEPSEEK_KEY")
        or os.getenv("COUNCIL_AGGREGATOR")
        or os.getenv("OPENROUTER_KEY")
        or openrouter_api_key
    )

    if not runtime_openrouter_api_key:
        raise ValueError(
            "OpenRouter API key missing. Set one of OPENROUTER_API_KEY, OPEN_ROUTER_DEEPSEEK_KEY, COUNCIL_AGGREGATOR, or OPENROUTER_KEY in your environment/.env."
        )

    headers = {
        "Authorization": f"Bearer {runtime_openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if openrouter_site_url:
        headers["HTTP-Referer"] = openrouter_site_url
    if openrouter_site_name:
        headers["X-OpenRouter-Title"] = openrouter_site_name

    payload = {
        "model": openrouter_model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=120
    )
    response.raise_for_status()
    body = response.json()

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError(f"Unexpected OpenRouter response format: {body}")

def call_gpt_oss_aggregator(recognition_id, utterance, context, profile, sentiment, dynamics, emotional_shift):
    """Backward-compatible wrapper around the new OpenRouter DeepSeek aggregator."""
    return call_council_aggregator(
        recognition_id,
        utterance,
        context,
        profile,
        sentiment,
        dynamics,
        emotional_shift
    )

# --- 5. CORE LOGIC ---

# def run_phase2_council(scene_data):
#     # -- LEVEL 1: GLOBAL CONTEXT --
#     global_context = groq_llm_call(prompt=f"{context_manager_prompt}\n\nScene Data: {scene_data}", 
#                                    model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=context_manager_api)
#     social_graph = groq_llm_call(prompt=f"{relational_graph_prompt}\n\nScene Data: {scene_data}", 
#                                   model="meta-llama/llama-4-maverick-17b-128e-instruct", api_key=relational_graph_api)

#     results = []
#     # -- LEVEL 2: THE SPECIALISTS (Parallel) --
#     for utterance in scene_data:
#         # Boosted workers for your powerful laptop
#         with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#             f3 = executor.submit(call_tuned_profiler, utterance, social_graph)
#             f4 = executor.submit(call_tuned_sentiment, utterance)
            
#             profile_report = f3.result()
#             sentiment_report = f4.result()

#         # Agent 5: Social Dynamics Expert
#         dynamics_report = call_social_dynamics(utterance, profile_report, social_graph)

#         # -- LEVEL 3: THE FINAL VERDICT --
#         final_prediction = call_gpt_oss_aggregator(
#             u['Recognition_ID'], utterance, global_context, profile_report, sentiment_report, dynamics_report
#         )
#         results.append(final_prediction)
    
#     return results

if __name__ == "__main__":
    try:
        # Test the connection to the fine-tuned endpoint
        response = profiler_model.generate_content("Test: Who is Chandler Bing?")
        print("✅ Connection Successful!")
        print(f"Sample Output: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")