import subprocess
import os

from dotenv import load_dotenv

load_dotenv()

import logging

# module logger (configured in __main__)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_LOCATION = os.getenv("DEFAULT_MODEL_LOCATION")


def get_access_token() -> str:
    """Fetches the short-lived access token using gcloud CLI."""
    try:
        # Executes the successful gcloud auth command
        token = (
            subprocess.check_output(["gcloud", "auth", "print-access-token"])
            .strip()
            .decode("utf-8")
        )
        return token
    except subprocess.CalledProcessError as e:
        logger.error(
            "Error fetching token: gcloud CLI may not be installed or authenticated."
        )
        raise Exception(
            "Authentication Error. Run 'gcloud auth application-default login'."
        ) from e


def get_model_info_from_env(model_key: str):
    """Resolve a model's MaaS name and location from environment variables.

    Expects env vars like <PREFIX>_MODEL_NAME and <PREFIX>_MODEL_LOCATION where
    PREFIX is the uppercase form of `model_key` (underscores instead of dashes).
    Falls back to global `MODEL_LOCATION` if per-model location is not set.
    """
    prefix = model_key.upper().replace("-", "_").replace(".", "_")
    model_name = os.getenv(f"{prefix}_MODEL_NAME")
    model_location = os.getenv(f"{prefix}_MODEL_LOCATION") or DEFAULT_MODEL_LOCATION
    return model_name, model_location
