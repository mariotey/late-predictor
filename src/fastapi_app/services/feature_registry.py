import json
import logging
import utils.supabase_client as supabase_client
from utils.logger import setup_logging
from config import FEATURE_REGISTRY_CONFIG_COL, FEATURE_REGISTRY_OUTPUT_PATH

setup_logging()
logger = logging.getLogger(__name__)


def refresh_feature_registry():
    registry_config = supabase_client.get_latest_registry()[FEATURE_REGISTRY_CONFIG_COL]

    with open(FEATURE_REGISTRY_OUTPUT_PATH, "w") as f:
        json.dump(registry_config, f, indent=2)

    logger.info("📦 Feature registry refreshed")


def load_feature_registry():
    with open(FEATURE_REGISTRY_OUTPUT_PATH, "r") as f:
        return json.load(f)