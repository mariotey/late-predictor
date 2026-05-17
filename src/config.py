"""Configuration Constants"""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent

# Data URL Path
RAW_DATA_PATH: str = "https://docs.google.com/spreadsheets/d/1-oE6cmsbq8TFLB7tVy6uIJnyM1Ab3fIaJ7_wLImESb4/export?format=csv"

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

# Directory Paths
ETL_GOOGLESHEET_DIR: Path = REPO_ROOT / "etl_googlesheet"
FEATURE_ENGINEERING_DIR: Path = REPO_ROOT / "feature_engineering"
FASTAPI_DIR: Path = REPO_ROOT / "fastapi_app"
FASTAPI_MODELS_DIR: Path = FASTAPI_DIR / "models"
FASTAPI_MODEL_ARTIFACT_DIR: Path = FASTAPI_MODELS_DIR / "artifacts"

# Feature Registry Path
FEATURE_REGISTRY_INPUT_PATH: Path = FEATURE_ENGINEERING_DIR / "feature_registry.yaml"
FEATURE_REGISTRY_OUTPUT_PATH: Path = FASTAPI_MODELS_DIR / "feature_registry.json"

# Model Artifact Paths
TRAINED_MODELS_PATH: Path = FASTAPI_MODEL_ARTIFACT_DIR / "trained_models.pkl"
TOP_MODELS_PATH: Path = FASTAPI_MODEL_ARTIFACT_DIR / "top_models.pkl"
ONEHOT_COL_PATH: Path = FASTAPI_MODEL_ARTIFACT_DIR / "onehot_columns.pkl"

# Model Hyperparameter
ENSEMBLE_NUM: int = 2
