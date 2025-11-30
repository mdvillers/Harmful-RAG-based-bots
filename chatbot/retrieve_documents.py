import os
from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
import logging

"""Retrieval helpers for RAG experiments.

This module provides a small wrapper around Google Discovery Engine search
used by the experimental RAG pipeline. The main exported function,
``retrieve_faq_answer(user_query: str) -> str``, performs a search using the
configured Discovery Engine data store and returns human-readable context
formatted as question/answer text that can be concatenated with a prompt
and passed to an LLM.

Key behavior and assumptions:
- Reads configuration from environment variables: ``PROJECT_ID``,
    ``SEARCH_REGION`` (used to build the endpoint), and ``DATA_STORE_ID``.
- Builds a ``SearchServiceClient`` with the regional endpoint and issues a
    ``SearchRequest`` (currently using ``page_size=1`` and ``serving_config="default_config"``).
- Collects the retrieved document's ``question_text`` and its ``answers``
    (each expected to contain ``answer_text``), formats them as plain text
    (``Q: ...`` and ``A: ...`` lines) and returns the combined string.

Usage notes for experiments:
- The returned string is intended to be prepended or inserted into an LLM
    prompt as grounding/context for RAG evaluation runs.
- If you need more context, increase ``page_size`` or change the
    ``serving_config`` name to match a different Discovery Engine config.
- The function logs debug information about the search configuration; set
    the process log level to ``DEBUG`` to see internal values during runs.

This file is used by the higher-level experiment runner (e.g. ``chatbot/main.py``)
to retrieve evidence for queries when running model evaluation or adversarial
prompting experiments.
"""

# module logger (configured in __main__)
logger = logging.getLogger(__name__)

load_dotenv()


# --- CONFIGURATION ---
PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_LOCATION = os.getenv("MODEL_LOCATION")
SEARCH_LOCATION = os.getenv("SEARCH_REGION")
DATA_STORE_ID = os.getenv("DATA_STORE_ID")


# --- 1. RETRIEVAL FUNCTION (Confirmed Working) ---
def retrieve_faq_answer(user_query: str) -> str:
    # Set the correct US multi-region endpoint for Discovery Engine
    client_options = ClientOptions(api_endpoint=f"{SEARCH_LOCATION}-discoveryengine.googleapis.com")
    client = discoveryengine.SearchServiceClient(client_options=client_options)

    serving_config = client.serving_config_path(
        project=PROJECT_ID,
        location=SEARCH_LOCATION,
        data_store=DATA_STORE_ID,
        serving_config="default_config",
    )

    logger.debug("Discovery SEARCH_LOCATION: %s", SEARCH_LOCATION)
    logger.debug("Discovery DATA_STORE_ID: %s", DATA_STORE_ID)

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=user_query,
        page_size=1,
    )

    response = client.search(request)
    retrieved_knowledge = []

    for result in response.results:
        faq_entry = result.document.struct_data
        question = faq_entry.get("question_text")
        answers = faq_entry.get("answers")
        answer_texts = []

        for answer in answers:
            answer_text = answer.get("answer_text")
            answer_texts.append("A: " + answer_text)

        combined_answer = "\n".join(answer_texts)
        formatted_knowledge = f"Q: {question}\n{combined_answer}"
        retrieved_knowledge.append(formatted_knowledge)

    return "\n\n".join(retrieved_knowledge)
