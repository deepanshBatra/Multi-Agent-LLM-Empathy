from numpy import character
import vertexai
from vertexai.generative_models import GenerativeModel
import json
import concurrent.futures
from agents import *

# 1. Initialize with your real project ID
vertexai.init(project='project-77549f95-0391-4b29-911', location='us-central1')

# 2. Agent 3: Character Profiler 
# Use the numerical Model ID from your registry
profiler_model = GenerativeModel(
    "projects/project-77549f95-0391-4b29-911/locations/us-central1/models/2352478794905812992"
)

# 3. Agent 4: Sentiment Analyzer
sentiment_model = GenerativeModel(
    "projects/project-77549f95-0391-4b29-911/locations/us-central1/models/7742443123938164736"
)

context_manager_api = os.getenv("CONTEXT_MANAGER") # groq
relational_graph_api = os.getenv("RELATIONAL_GRAPH_MANAGER") # groq
social_dynamics_api = os.getenv("SOCIAL_DYNAMICS_EXPERT") # gemini
aggregator_api = os.getenv("COUNCIL_AGGREGATOR") # groq

# Get the absolute path to the directory where THIS script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prompts_path = os.path.join(BASE_DIR, "prompts")

# Load prompts using cross-platform safe paths
context_manager_prompt = load_prompt(os.path.join(prompts_path, "context_manager.txt"))
social_dynamics_expert_prompt = load_prompt(os.path.join(prompts_path, "social_dynamics_expert.txt"))
empathy_reasonar_prompt = load_prompt(os.path.join(prompts_path, "empathy_reasonar.txt"))
council_aggregator_prompt = load_prompt(os.path.join(prompts_path, "council_aggregator.txt"))
relational_graph_prompt = load_prompt(os.path.join(prompts_path, "relational_graph.txt"))
character_profiler_prompt = load_prompt(os.path.join(prompts_path, "character_profiler.txt"))

# Helper for Tuned Gemini Agents
def call_tuned_gemini(model_obj, prompt, utterance_text):
    """General caller for your fine-tuned Gemini endpoints."""
    # We combine the system-style prompt with the target utterance
    full_prompt = f"{prompt}\n\nTARGET UTTERANCE: {utterance_text}"
    response = model_obj.generate_content(full_prompt)
    return response.text

# specific functions for your loop
def call_tuned_profiler(utterance, social_graph):
    # Pass the social graph so the profiler knows the context
    context = f"Social Context: {social_graph}"
    return call_tuned_gemini(profiler_model, character_profiler_prompt, f"{context}\nUtterance: {utterance}")

def call_tuned_sentiment(utterance):
    return call_tuned_gemini(sentiment_model, empathy_reasonar_prompt, utterance)

def call_social_dynamics(utterance, profile, social_graph):
    """Agent 5: Synthesizes relational data and character DNA."""
    prompt = f"""
    {social_dynamics_expert_prompt}
    
    UTTERANCE: {utterance}
    CHARACTER PROFILE: {profile}
    RELATIONAL GRAPH: {social_graph}
    """
    # Using Maverick for high-reasoning social bridge
    return gemini_llm_call(prompt=prompt, api_key=social_dynamics_api)

def call_gpt_oss_aggregator(utterance, context, profile, sentiment, dynamics):
    """Agent 6: The Final Judge."""
    prompt = f"""
    {council_aggregator_prompt}
    
    TARGET UTTERANCE: {utterance}
    SCENE CONTEXT: {context}
    
    REPORTS:
    1. Character Profile: {profile}
    2. Sentiment Scores: {sentiment}
    3. Social Dynamics: {dynamics}
    """
    # GPT-OSS 120B is excellent at weighing conflicting reports
    return groq_llm_call(prompt=prompt, model="openai/gpt-oss-120b", api_key=aggregator_api)

def run_phase2_council(scene_data):
    """
    scene_data: A list of utterances in a Dialogue_ID
    """
    # -- LEVEL 1: GLOBAL CONTEXT --
    # Agent 1 & 2 run first to set the stage
    global_context = groq_llm_call(prompt=f"{context_manager_prompt}\n\nScene Data: {scene_data}", model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=context_manager_api)
    social_graph = groq_llm_call(prompt=f"{relational_graph_prompt}\n\nScene Data: {scene_data}", model="meta-llama/llama-4-maverick-17b-128e-instruct", api_key=relational_graph_api)

    # -- LEVEL 2: THE SPECIALISTS (Parallel) --
    # These agents analyze the target utterance using the Level 1 context
    results = []
    for utterance in scene_data:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Agent 3: Fine-tuned Character Profiler
            f3 = executor.submit(call_tuned_profiler, utterance, social_graph)
            # Agent 4: Fine-tuned Sentiment Analyst
            f4 = executor.submit(call_tuned_sentiment, utterance)
            
            profile_report = f3.result()
            sentiment_report = f4.result()

        # Agent 5: Social Dynamics Expert (Reasoning Bridge)
        dynamics_report = call_social_dynamics(utterance, profile_report, social_graph)

        # -- LEVEL 3: THE FINAL VERDICT --
        # Agent 6: Aggregator
        final_prediction = call_gpt_oss_aggregator(
            utterance, global_context, profile_report, sentiment_report, dynamics_report
        )
        results.append(final_prediction)
    
    return results