import os
import json
import concurrent.futures
import vertexai
from vertexai.generative_models import GenerativeModel
from google import genai
from src.agents import * # Assuming your groq_llm_call and load_prompt are defined here

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
aggregator_api = os.getenv("COUNCIL_AGGREGATOR")

# --- 3. PROMPT LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prompts_path = os.path.join(BASE_DIR, "prompts")

context_manager_prompt = load_prompt(os.path.join(prompts_path, "context_manager.txt"))
social_dynamics_expert_prompt = load_prompt(os.path.join(prompts_path, "social_dynamics_expert.txt"))
empathy_reasonar_prompt = load_prompt(os.path.join(prompts_path, "empathy_reasonar.txt"))
council_aggregator_prompt = load_prompt(os.path.join(prompts_path, "council_aggregator.txt"))
relational_graph_prompt = load_prompt(os.path.join(prompts_path, "relational_graph.txt"))
character_profiler_prompt = load_prompt(os.path.join(prompts_path, "character_profiler.txt"))

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

def call_gpt_oss_aggregator(recognition_id, utterance, context, profile, sentiment, dynamics):
    """Agent 6: Synthesizes expert reports into a final MELD label."""
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
    """
    return groq_llm_call(prompt=prompt, model="openai/gpt-oss-120b", api_key=aggregator_api)

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