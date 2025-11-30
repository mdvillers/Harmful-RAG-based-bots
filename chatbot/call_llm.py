import os
import time
import random
from typing import Dict
from openai import OpenAI

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    before_sleep_log,
)

import logging

logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


PROJECT_ID = os.getenv("PROJECT_ID")

SYSTEM_PROMPT = (
    "You are a technical support bot. Use only the context provided. Be concise."
)


# Retry configuration (can be overridden with env vars)
MAX_RETRIES = int(os.getenv("RETRY_MAX_ATTEMPTS", "10"))
BASE_BACKOFF = float(os.getenv("RETRY_BASE_DELAY", "5"))  # seconds
MAX_BACKOFF = float(os.getenv("RETRY_MAX_DELAY", "60"))  # seconds


def _is_transient_error(exc: Exception) -> bool:
    """Heuristic to decide whether an exception is transient (rate/5xx).

    We inspect common text patterns and known attributes (status codes, headers)
    where available. This is intentionally conservative.
    """
    try:
        msg = str(exc).lower()
    except Exception:
        msg = ""

    # common transient signals
    if "rate limit" in msg or "rate_limit" in msg or "too many requests" in msg or "429" in msg:
        return True
    if "503" in msg or "500" in msg or "server error" in msg or "service unavailable" in msg:
        return True

    # Some HTTP client exceptions expose a 'status_code' attribute
    status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        if status_code is not None:
            code = int(status_code)
            if code == 429 or 500 <= code < 600:
                return True
    except Exception:
        pass

    return False


def _extract_retry_after(exc: Exception) -> float:
    """Try to extract a Retry-After value (seconds) from the exception, if present."""
    headers = getattr(exc, "headers", None) or getattr(exc, "response", None)
    if isinstance(headers, dict):
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra:
            try:
                return float(ra)
            except Exception:
                pass
    # Some exception objects may have .response.headers
    resp = getattr(exc, "response", None)
    if resp is not None:
        h = getattr(resp, "headers", None)
        if isinstance(h, dict):
            ra = h.get("Retry-After") or h.get("retry-after")
            if ra:
                try:
                    return float(ra)
                except Exception:
                    pass
    return 0.0


class _WaitWithRetryAfter:
    """Custom tenacity wait strategy that applies exponential backoff with jitter
    and respects a Retry-After header when present on the last exception.
    """

    def __init__(self, base: float = BASE_BACKOFF, max_wait: float = MAX_BACKOFF):
        self.base = base
        self.max_wait = max_wait

    def __call__(self, retry_state) -> float:
        # attempt_number starts at 1
        attempt_no = retry_state.attempt_number
        backoff_base = min(self.max_wait, self.base * (2 ** (attempt_no - 1)))
        sleep_for = random.uniform(0, backoff_base)

        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc is not None:
            try:
                ra = _extract_retry_after(exc)
                if ra and ra > sleep_for:
                    sleep_for = ra
            except Exception:
                pass

        return sleep_for


def ask_llm_openai_compatible(
    model_name: str, prompt: str, location: str, access_token: str, use_system_prompt: bool = True
) -> Dict[str, str]:
    """Uses the OpenAI client to hit the specific MaaS gateway endpoint (for DeepSeek, etc.).

    Implements a tenacity-based retry with exponential backoff + jitter for
    transient errors. The retry decision uses ``_is_transient_error``.
    """
    # Check if the model is Gemini and requires non-thinking mode
    is_gemini_flash = "gemini" in model_name.lower() and "flash" in model_name.lower()
    if is_gemini_flash:
        logger.info("Using Gemini Flash model; applying non-thinking mode settings.")

    OPENAPI_BASE_URL = f"https://{location}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{location}/endpoints/openapi"

    logger.debug("Project ID: %s", PROJECT_ID)
    logger.info("Using OpenAI-compatible client, URL: %s", OPENAPI_BASE_URL)

    client = OpenAI(base_url=OPENAPI_BASE_URL, api_key=access_token)

    extra_config = {}
    if is_gemini_flash:
        extra_config["extra_body"] = {"google": {"thinking_config": {"thinking_budget": 0}}}
        logger.info(
            "-> Activating NON-THINKING FAST MODE (thinkingBudget: 0) for Gemini Flash."
        )

    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    @retry(
        retry=retry_if_exception(_is_transient_error),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=_WaitWithRetryAfter(BASE_BACKOFF, MAX_BACKOFF),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_api():
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            extra_body=extra_config,
        )

    try:
        response = _call_api()
        return {
            "model": model_name,
            "answer": response.choices[0].message.content.strip(),
        }
    except Exception as e:
        logger.exception("Failed to generate content from model %s after retries", model_name)
        return {
            "model": model_name,
            "answer": f"ERROR: Failed to generate content via OpenAI gateway. Details: {e}",
        }
    