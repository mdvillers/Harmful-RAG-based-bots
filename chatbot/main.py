import os
from dotenv import load_dotenv
import argparse
import logging
import datetime
from logging.handlers import RotatingFileHandler
from experiment import start_experiment

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





# --- 3. MAIN RAG LOOP EXECUTION ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="RAG chatbot runner: choose model via --model and optionally override query"
    )
    parser.add_argument(
        "--target-model",
        required=True,
        help="Target model key name to use for generation (e.g. llama4, mistral_small)",
    )
    parser.add_argument(
        "--verifier-model",
        required=True,
        help="Verifier model key name to use for classifying responses (e.g. verifier_small)",
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
        target_model=args.target_model,
        verifier_model=args.verifier_model,
        output_file=args.output_file,
        use_system_prompt=use_system_prompt,
        num_queries=args.num_queries,
    )


if __name__ == "__main__":
    main()
