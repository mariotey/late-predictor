"""Configuration Constants"""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent

# Data Table Names
CLEAN_DATA_NAME: str = "clean_data"
FEATURE_STORE_NAME: str = "feature_store"
FEATURE_REGISTRY_NAME: str = "feature_registry"
FEEDBACK_STORE_NAME: str = "feedback_store"

# Data Table Column Names
FEATURE_REGISTRY_CONFIG_COL: str = "config"

# Model Artifact Paths
MODELS_DIR: Path = REPO_ROOT / "models"
MODEL_ARTIFACT_DIR: Path = MODELS_DIR / "artifacts"
TRAINED_MODELS_PATH: Path = MODEL_ARTIFACT_DIR / "trained_models.pkl"
TOP_MODELS_PATH: Path = MODEL_ARTIFACT_DIR / "top_models.pkl"
ONEHOT_COL_PATH: Path = MODEL_ARTIFACT_DIR / "onehot_columns.pkl"
FEATURE_REGISTRY_PATH: Path = MODELS_DIR / "feature_registry.json"

# Model Hyperparameter
ENSEMBLE_NUM: int = 2
