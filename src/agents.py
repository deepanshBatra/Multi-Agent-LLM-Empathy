from json import load
from groq import Groq
from google import genai
import os
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLAMA_3_API_KEY = os.getenv("LLAMA_3.3_API_KEY")


def gemini_llm_call(prompt, model="gemini-3-flash-preview", api_key=GEMINI_API_KEY):
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    return response.text


def groq_llm_call(prompt, model, api_key):
    client = Groq(api_key=api_key)
    params = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
        "max_completion_tokens": 8192,
        "stream": False, # Changed to False
    }
    chat_completion = client.chat.completions.create(
      messages=[
          {
              "role": "user",
              "content": prompt,
          }
      ],
      model=model,
      temperature=params["temperature"],
      max_tokens=params["max_completion_tokens"]
  )
    return chat_completion.choices[0].message.content

 
def load_prompt(file_address):
    with open(file_address, "r") as f:
        return f.read()
    

def run_multi_agent_conversation(context_dict):
    prompts_path = "prompts" 

    context_manager_prompt = load_prompt(os.path.join(prompts_path, "context_manager.txt"))
    social_dynamics_expert_prompt = load_prompt(os.path.join(prompts_path, "social_dynamics_expert.txt"))
    empathy_reasonar_prompt = load_prompt(os.path.join(prompts_path, "empathy_reasonar.txt"))
    council_aggregator_prompt = load_prompt(os.path.join(prompts_path, "council_aggregator.txt"))

    # 2. Level 1: Context Manager (Historian) - MUST RUN FIRST
    context_manager_agent_response = groq_llm_call(
        prompt=f"{context_manager_prompt}\n\nRAW DATA:\n{context_dict}", 
        model="llama-3.3-70b-versatile", 
        api_key=LLAMA_3_API_KEY
    )

    # 3. Level 2: Specialists (Run in PARALLEL)
    # We define the specific calls for the threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both specialists to the thread pool
        future_social = executor.submit(
            groq_llm_call, 
            f"{social_dynamics_expert_prompt}\n\nDATA:\n{context_dict}\n\nANALYSIS:\n{context_manager_agent_response}",
            "llama-3.1-8b-instant",
            LLAMA_API_KEY
        )
        
        future_empathy = executor.submit(
            groq_llm_call, 
            f"{empathy_reasonar_prompt}\n\nDATA:\n{context_dict}\n\nANALYSIS:\n{context_manager_agent_response}",
            "qwen/qwen3-32b",
            DEEPSEEK_API_KEY
        )

        # Retrieve results as they finish
        social_dynamics_agent_response = future_social.result()
        empathy_reasonar_agent_response = future_empathy.result()

    # 4. Level 3: Aggregator (The Judge) - MUST RUN LAST
    aggregator_final_prompt = f"""
    {council_aggregator_prompt}
    
    CRITICAL METADATA:
    RECOGNITION_ID: {context_dict['recognition_id']}

    EXPERT REPORTS:
    1. Context Manager: {context_manager_agent_response}
    2. Social Dynamics Expert: {social_dynamics_agent_response}
    3. Empathy Reasoner: {empathy_reasonar_agent_response}
    """

    return groq_llm_call(
        prompt=aggregator_final_prompt, 
        model="openai/gpt-oss-120b",
        api_key=GPT_OSS_API_KEY
    )
