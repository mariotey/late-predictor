"""Configuration Constants"""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent

# Data Table Names
APPT_NAME: str = "Appointment"
CATEGORY_NAME: str = "Category"
FEATURES_NAME: str = "Features"
FEATURE_REGISTRY_NAME: str = "FeatureRegistry"
FEEDBACK_NAME: str = "Feedback"

# Data Table Column Names
APPT_ID_COL: str = "appt_id"
CATEGORY_ID_COL: str = "category_id"
FEATURES_ID_COL: str = "feature_id"
FEATURE_REGISTRY_ID_COL: str = "f_reg_id"
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
