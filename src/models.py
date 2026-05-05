from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

LINEAR_MODELS = [
    ("linear_regression", LinearRegression()),
    ("ridge", Ridge(alpha=1.0)),
]

TREE_MODELS = [
    ("random_forest", RandomForestRegressor()),
    ("gboost", GradientBoostingRegressor()),
]