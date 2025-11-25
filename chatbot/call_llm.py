import os
from typing import Dict
from openai import OpenAI
from utils import get_access_token

import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


PROJECT_ID = os.getenv("PROJECT_ID")


# --- 2. GENERATION FUNCTION ---
def ask_llm_openai_compatible(
    model_name: str, prompt: str, location: str
) -> Dict[str, str]:
    """Uses the OpenAI client to hit the specific MaaS gateway endpoint (for DeepSeek, etc.)."""

    # 1. Get the Bearer Token
    access_token = get_access_token()
    if not access_token:
        raise Exception("Authentication failure for OpenAI-compatible client.")

    # 2. Configure the Base URL to the regional OpenAPI gateway
    # This URL mirrors the successful curl command structure
    # https://${ENDPOINT}/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/openapi/chat/completions \

    OPENAPI_BASE_URL = f"https://{location}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{location}/endpoints/openapi"

    logger.debug("Project ID: %s", PROJECT_ID)
    logger.info("Using OpenAI-compatible client, URL: %s", OPENAPI_BASE_URL)

    client = OpenAI(
        base_url=OPENAPI_BASE_URL,
        api_key=access_token,  # Pass the Bearer token as the API key
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a technical support bot. Be concise.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return {
        "model": model_name,
        "answer": response.choices[0].message.content.strip(),
    }
