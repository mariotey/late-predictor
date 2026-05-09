import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def Cat_LabelEncoding(df, cols):
    logger.info(f"Applying Label Encoding on columns: {cols}\n")

    modified_df = df.copy()
    le = LabelEncoder()

    for col in cols:
        before_nunique = modified_df[col].nunique()
        modified_df[col] = le.fit_transform(modified_df[col])

        logger.info(
            f"Encoded '{col}' | unique values before={before_nunique}, "
            f"after={modified_df[col].nunique()}\n"
        )

    logger.info(f"[LabelEncoding] Final shape: {modified_df.shape}\n")
    logger.info(f"[LabelEncoding] Columns: {modified_df.columns}\n")

    return modified_df


def Cat_OneHotEncoding(df, cols):
    logger.info(f"Applying OneHot Encoding on columns: {cols}\n")

    before_shape = df.shape
    modified_df = pd.get_dummies(df, columns=cols)

    logger.info(
        f"OneHot encoding complete | before_shape={before_shape}, "
        f"after_shape={modified_df.shape}, "
        f"new_columns_added={modified_df.shape[1] - before_shape[1]}\n"
    )

    logger.info(f"[OneHotEncoding] Final shape: {modified_df.shape}\n")
    logger.info(f"[OneHotEncoding] Columns: {modified_df.columns}\n")

    return modified_df