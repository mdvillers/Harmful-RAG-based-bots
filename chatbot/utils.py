import subprocess
import os
import platform
import time
from dotenv import load_dotenv

load_dotenv()

import logging

# module logger (configured in __main__)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_LOCATION = os.getenv("DEFAULT_MODEL_LOCATION")

# Simple in-process cache for the access token. We know the token expires in 3600s.
_cached_token = None
_cached_at = 0.0
_TOKEN_TTL = 3600  # seconds


def get_access_token(force_refresh: bool = False) -> str:
    """Fetches the short-lived access token using gcloud CLI.

    Caches the token in-process for `_TOKEN_TTL` seconds. If `force_refresh`
    is True or the cached token is expired, fetches a new token from gcloud.
    """
    global _cached_token, _cached_at
    now = time.time()

    if not force_refresh and _cached_token and (now - _cached_at) < (_TOKEN_TTL - 60):
        logger.debug("Using cached access token (age=%.0fs)", now - _cached_at)
        return _cached_token

    try:
        # Choose command depending on OS
        if platform.system().lower().startswith("win"):
            cmd = ["cmd.exe", "/c", "gcloud", "auth", "print-access-token"]
        else:
            cmd = ["gcloud", "auth", "print-access-token"]

        logger.debug("Running token command: %s", cmd)

        token = subprocess.check_output(cmd).strip().decode("utf-8")
        _cached_token = token
        _cached_at = time.time()
        logger.debug("Fetched new access token and cached it")
        return token
    except FileNotFoundError as e:
        logger.error("gcloud CLI not found: %s", e)
        raise Exception("gcloud CLI not found. Install and authenticate with gcloud.") from e
    except subprocess.CalledProcessError as e:
        logger.error(
            "Error fetching token: gcloud CLI may not be installed or authenticated. Return code: %s", e.returncode
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
