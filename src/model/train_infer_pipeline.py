"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""
import logging
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score
)

import xgboost as xgb
import matplotlib.pyplot as plt

# ===============================
# Project paths
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Logging
# ===============================
LOG_PATH = ARTIFACTS_DIR / "train_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ===============================
# Config
# ===============================
TARGET_COL = "target_bin"
RANDOM_STATE = 42

BEST_THRESHOLD = 0.39 

XGB_PARAMS = {
        "n_estimators": 262,
        "learning_rate": 0.06985405444530932,
        "max_depth": 9,
        "min_child_weight": 3,
        "subsample": 0.6290777379947342,
        "colsample_bytree": 0.6199838754923778,
        "reg_alpha": 0.3077574084366931,
        "reg_lambda": 0.3555228636422837,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",  
        "random_state": RANDOM_STATE,
        "n_jobs": -1
}

# ===============================
# Metrics
# ===============================
def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "recall_used": recall_score(y_true, y_pred, pos_label=1),
        "precision_used": precision_score(y_true, y_pred, pos_label=1)
    }

# ===============================
# Feature importance plot
# ===============================
def plot_feature_importance(model, feature_names, out_path, top_n=10):
    booster = model.get_booster()
    scores = booster.get_score(importance_type="gain")

    df_imp = (
        pd.DataFrame(scores.items(), columns=["feature", "gain"])
        .sort_values("gain", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(8, 5))
    plt.barh(df_imp["feature"], df_imp["gain"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance (Gain)")
    plt.xlabel("Gain")
    plt.tight_layout()

    plt.savefig(out_path)
    plt.close()

# ===============================
# Train & Infer pipeline
# ===============================
def main():
    logger.info("Starting FINAL TRAINING PIPELINE")

    # --------------------------------------------------
    # Step 1 & 2 are assumed already executed:
    # - build_dataset.py
    # - feat_eng_pipeline.py
    # --------------------------------------------------

    train_path = DATA_PROCESSED_DIR / "train_fe.parquet"
    test_path = DATA_PROCESSED_DIR / "test_fe.parquet"

    logger.info("Loading FE datasets")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Test shape: {X_test.shape}")

    # --------------------------------------------------
    # Step 3: Train final XGBoost
    # --------------------------------------------------
    logger.info("Training final XGBoost model")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    model_path = ARTIFACTS_DIR / "xgb_final_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    logger.info(f"Model saved at {model_path}")

    # Feature importance
    fi_path = ARTIFACTS_DIR / "xgb_feature_importance_top10.png"
    plot_feature_importance(
        model=model,
        feature_names=X_train.columns,
        out_path=fi_path
    )

    logger.info(f"Feature importance saved at {fi_path}")

    # --------------------------------------------------
    # Step 4: Inference + metrics
    # --------------------------------------------------
    logger.info("Running inference on test set")

    y_test_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_test_proba, BEST_THRESHOLD)

    logger.info("Final TEST metrics")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    metrics_df = pd.DataFrame([metrics])
    metrics_path = ARTIFACTS_DIR / "final_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    logger.info(f"Metrics saved at {metrics_path}")
    logger.info("FINAL PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()    

