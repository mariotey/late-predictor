"""Configuration Constants"""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent

# Google Sheet URL Path
DATA_URL: str = "https://docs.google.com/spreadsheets/d/1-oE6cmsbq8TFLB7tVy6uIJnyM1Ab3fIaJ7_wLImESb4/export?format=csv"

# Data Paths
DATA_DIR: Path = REPO_ROOT / "data"
CLEAN_DATA_PATH: Path = DATA_DIR / "feature_df.parquet"

# Model Artifact Paths
MODELS_DIR: Path = REPO_ROOT / "models"
MODEL_ARTIFACT_DIR: Path = MODELS_DIR / "artifacts"
TRAINED_MODELS_PATH: Path = MODEL_ARTIFACT_DIR / "trained_models.pkl"
TOP_MODELS_PATH: Path = MODEL_ARTIFACT_DIR / "top_models.pkl"

# Model Hyperparameter
ENSEMBLE_NUM: int = 2
