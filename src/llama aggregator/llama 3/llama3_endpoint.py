import os

import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel

load_dotenv()

# Google Cloud / Vertex AI configuration
PROJECT_ID = os.getenv("LLAMA_MODEL_PROJECT_ID") or os.getenv("TUNED_MODEL_PROJECT_ID")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
ENDPOINT_ID = "2346569469662330880"

if not PROJECT_ID:
    raise ValueError(
        "Missing project id. Set LLAMA_MODEL_PROJECT_ID or TUNED_MODEL_PROJECT_ID in your environment/.env."
    )

vertexai.init(project=PROJECT_ID, location=LOCATION)

llama3_model = GenerativeModel(
    f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
)


def call_llama3(prompt: str) -> str:
    """Call the deployed Vertex AI endpoint and return plain text output."""
    response = llama3_model.generate_content(prompt)
    return response.text


if __name__ == "__main__":
    test_prompt = os.getenv("LLAMA_TEST_PROMPT", "Test: Introduce yourself in one sentence.")
    try:
        output = call_llama3(test_prompt)
        print("Connection successful.")
        print(f"Response: {output[:300]}")
    except Exception as exc:
        print(f"Connection failed: {exc}")
