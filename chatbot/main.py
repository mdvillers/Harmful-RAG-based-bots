import os
import json
from dotenv import load_dotenv
import argparse
import logging
import random
import datetime
from logging.handlers import RotatingFileHandler

from utils import get_access_token, get_model_info_from_env
from retrieve_documents import retrieve_faq_answer
from call_llm import ask_llm_openai_compatible

# module logger (configured in __main__)
logger = logging.getLogger(__name__)

load_dotenv()

# Create the RAG prompt (the same for all models)
PROMPT_FOR_MODEL = {
    "query_before_context": """
                    You are a technical support bot. Be concise and helpful.
                    User Question: {user_query}

                    --- FAQ Context ---
                    {context}
                    -------------------
                    """,
    "context_before_query": """
                    You are a technical support bot for video games. 
                    Use only the provided FAQ Context to answer the user's question. 
                    If the context does not contain the answer, state that you cannot find the information. 
                    Be concise and helpful.

                    --- FAQ Context ---
                    {context}
                    -------------------
                    User Question: {user_query}
""",
}


def start_experiment(
    prompt_dataset_file: str,
    final_prompt: str,
    model: str,
    output_file: str,
    use_system_prompt: bool = True,
    num_queries: int = -1,
):
    """Placeholder for any experiment initialization logic."""
    logger.info("Experiment started.")

    # Resolve model id and per-model location from env
    model_name, model_location_for_model = get_model_info_from_env(model)
    if not model_name:
        logger.error(
            "No model id found for model key '%s'. Ensure env var %s_MODEL_ID is set.",
            model,
            model.upper().replace("-", "_").replace(".", "_"),
        )
        exit(2)

    # Log configuration details at DEBUG

    logger.debug("Selected model key: %s", model)
    logger.debug("Resolved model name: %s", model_name)
    logger.debug("Resolved model location: %s", model_location_for_model)

    # Get access token once
    logger.info("Fetching access token via gcloud...")
    access_token = get_access_token()
    logger.info("Access token retrieved.")

    # Read user queries from the prompt dataset jsonl file (one JSON object per line)
    with open(prompt_dataset_file, "r", encoding="utf-8") as f:
        lines = [l for l in (ln.strip() for ln in f) if l]

    total_lines = len(lines)
    logger.info("Prompt dataset contains %d entries", total_lines)

    if num_queries == -1:
        # process all serially in file order
        selected_lines = lines
        logger.info("Processing all entries serially (%d)", len(selected_lines))
    elif num_queries == 0:
        logger.info("--num-queries set to 0, nothing to process.")
        return
    else:
        # randomly pick num_queries unique lines
        k = min(num_queries, total_lines)
        selected_lines = random.sample(lines, k)
        logger.info("Processing %d randomly selected entries out of %d", k, total_lines)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for line in selected_lines:
            obj = None
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line in prompt dataset")
                continue
            question_text = obj.get("question_text")

            user_query = question_text

            logger.info("Processing user query: %s", user_query)

            logger.info("Starting RAG retrieval for user query")

            # 1. RETRIEVE CONTEXT
            context = retrieve_faq_answer(user_query)

            # Create the final RAG prompt
            logger.info("Retrieval complete; sending prompt to model")
            logger.debug("Retrieved context: %s", context)

            rag_prompt = final_prompt.format(context=context, user_query=user_query)
            logger.debug("RAG prompt length: %d characters", len(rag_prompt))

            # 2. GENERATE ANSWER using the selected model
            logger.info("Sending prompt to model %s", model_name)
            response = ask_llm_openai_compatible(
                model_name,
                rag_prompt,
                model_location_for_model,
                access_token=access_token,
                use_system_prompt=use_system_prompt,
            )
            logger.info("Generation complete for model %s", response.get("model"))
            logger.debug("Model response: %s", response.get("answer"))

            # save response in the jsonl again
            obj["model_response"] = response.get("answer")
            out_f.write(json.dumps(obj) + "\n")


# --- 3. MAIN RAG LOOP EXECUTION ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG chatbot runner: choose model via --model and optionally override query"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model key name to use (e.g. llama4, mistral_small, gemini_25_flash, deepseek_31)",
    )
    parser.add_argument(
        "--log-level",
        required=False,
        default="warn",
        help="Logging level (debug, info, warn, error). Default: warn",
    )

    parser.add_argument(
        "--prompt-dataset-file",
        required=True,
        help="Path to the JSONL file containing FAQ data with user queries",
    )

    parser.add_argument(
        "--num-queries",
        type=int,
        required=True,
        help=(
            "Number of queries to process from the prompt dataset. "
            "If -1 process all entries serially; if >0 randomly select that many entries."
        ),
    )

    parser.add_argument(
        "--prompt-format",
        choices=["query_before_context", "context_before_query"],
        default="query_before_context",
        help="RAG prompt format to use",
    )

    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        default=False,
        help="Whether to include the system prompt in the LLM call",
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to output JSONL file with model responses",
    )

    return parser.parse_args()


def ensure_output_dir(output_file: str) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def setup_logging(level_str: str) -> str:
    """Configure root logging with a console handler and rotating file handler.

    Returns the logfile path created (useful for info/debug messages).
    """
    level_name = level_str.upper()
    if level_name == "WARN":
        level_name = "WARNING"
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level '{level_str}', defaulting to WARNING")
        numeric_level = logging.WARNING

    # Ensure logs directory exists and create timestamped logfile
    logs_dir = os.path.join(os.getcwd(), "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        logs_dir = os.getcwd()

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logfile_path = os.path.join(logs_dir, f"{timestamp}.log")

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(logfile_path, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return logfile_path


def main():
    args = parse_args()

    # ensure output dir exists
    ensure_output_dir(args.output_file)

    logfile = setup_logging(args.log_level)
    logger.debug("Logging to file %s", logfile)

    use_system_prompt = not args.no_system_prompt
    logger.debug("Using system prompt: %s", use_system_prompt)

    final_prompt = PROMPT_FOR_MODEL[args.prompt_format]
    start_experiment(
        prompt_dataset_file=args.prompt_dataset_file,
        final_prompt=final_prompt,
        model=args.model,
        output_file=args.output_file,
        use_system_prompt=use_system_prompt,
        num_queries=args.num_queries,
    )


if __name__ == "__main__":
    main()
