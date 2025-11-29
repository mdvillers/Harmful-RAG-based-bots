import os
from typing import Dict
from openai import OpenAI
from utils import get_access_token

import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


PROJECT_ID = os.getenv("PROJECT_ID")

SYSTEM_PROMPT = (
    "You are a technical support bot. Use only the context provided. Be concise."
)


# --- 2. GENERATION FUNCTION ---
def ask_llm_openai_compatible(
    model_name: str, prompt: str, location: str
) -> Dict[str, str]:
    """Uses the OpenAI client to hit the specific MaaS gateway endpoint (for DeepSeek, etc.)."""
    # Check if the model is Gemini and requires non-thinking mode

    is_gemini_flash = "gemini" in model_name.lower() and "flash" in model_name.lower()
    if is_gemini_flash:
        logger.info("Using Gemini Flash model; applying non-thinking mode settings.")

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

    try:
        client = OpenAI(base_url=OPENAPI_BASE_URL, api_key=access_token)

        # 3. Setup Config (FOR GEMINI FAST MODE)
        # We use the 'extra_body' parameter to inject Gemini-specific configuration
        extra_config = {}
        if is_gemini_flash:
            # Setting thinkingBudget: 0 disables the complex reasoning for fast mode
            extra_config["extra_body"] = {"google":{"thinking_config": {"thinking_budget": 0}}}
            logger.info(
                "-> Activating NON-THINKING FAST MODE (thinkingBudget: 0) for Gemini Flash."
            )

        # 4. Execute the call
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
            # Pass the custom configuration through extra_body
            extra_body=extra_config,
        )

        return {
            "model": model_name,
            "answer": response.choices[0].message.content.strip(),
        }

    except Exception as e:
        logger.exception("Failed to generate content from model %s", model_name)
        return {
            "model": model_name,
            "answer": f"ERROR: Failed to generate content via OpenAI gateway. Details: {e}",
        }
