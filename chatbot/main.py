from dotenv import load_dotenv
import argparse
import logging

from utils import get_model_info_from_env
from retrieve_documents import retrieve_faq_answer
from call_llm import ask_llm_openai_compatible

# module logger (configured in __main__)
logger = logging.getLogger(__name__)

load_dotenv()

# --- 3. MAIN RAG LOOP EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG chatbot runner: choose model via --model and optionally override query"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key name to use (e.g. llama4, mistral_small, gemini_25_flash, deepseek_31)",
    )
    parser.add_argument(
        "--query",
        required=False,
        help="User query to run (if omitted a default example is used)",
    )
    parser.add_argument(
        "--log-level",
        required=False,
        default="warn",
        help="Logging level (debug, info, warn, error). Default: warn",
    )
    args = parser.parse_args()

    # Configure logging
    level_name = args.log_level.upper()
    # Accept 'WARN' as alias for WARNING
    if level_name == "WARN":
        level_name = "WARNING"
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level '{args.log_level}', defaulting to WARNING")
        numeric_level = logging.WARNING
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.setLevel(numeric_level)

    user_query = (
        args.query
        or "Will this game run on a Windows 7 machine, and what problems might I encounter?"
    )

    # Resolve model id and per-model location from env
    model_name, model_location_for_model = get_model_info_from_env(args.model)
    if not model_name:
        logger.error(
            "No model id found for model key '%s'. Ensure env var %s_MODEL_ID is set.",
            args.model,
            args.model.upper().replace("-", "_").replace(".", "_"),
        )
        exit(2)

    # Log configuration details at DEBUG

    logger.debug("Selected model key: %s", args.model)
    logger.debug("Resolved model name: %s", model_name)
    logger.debug("Resolved model location: %s", model_location_for_model)

    # 1. RETRIEVE CONTEXT
    context = retrieve_faq_answer(user_query)

    # Create the RAG prompt (the same for all models)
    rag_prompt = f"""
    You are a technical support bot for video games. Use only the provided FAQ Context to answer the user's question. 
    If the context does not contain the answer, state that you cannot find the information. Be concise and helpful.

    --- FAQ Context ---
    {context}
    -------------------

    User Question: {user_query}
    """

    logger.info("Retrieval complete; sending prompt to model")
    logger.debug("Retrieved context: %s", context)
    logger.debug("RAG prompt length: %d characters", len(rag_prompt))

    # 2. GENERATE ANSWER using the selected model
    response = ask_llm_openai_compatible(
        model_name, rag_prompt, model_location_for_model
    )
    logger.info("Generation complete for model %s", response.get("model"))
    logger.debug("Model response: %s", response.get("answer"))
    print(f"\n--- GENERATION: {response['model']} ---")
    print(f"Answer: {response['answer']}")
