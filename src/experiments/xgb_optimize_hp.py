"""
Este script busca optimizar los hiperparametros de un modelo XGBoost utilizando Optuna con validación cruzada estratificada.
XGBoost se selecciona como el modelo ganador el cual sera optimizado porque presenta el mejor equilibrio entre 
desempeño predictivo y generalización frente a las alternativas evaluadas.
En validación cruzada, XGBoost alcanza:

- la mayor accuracy (0.866), superando el umbral requerido de 0.86,
- el mejor ROC-AUC (0.943), indicando una superior capacidad de discriminación entre productos nuevos y usados,
- y el mayor recall (0.879) para la clase used, que es la de mayor interés de negocio.

En conjunto, estos resultados indican que XGBoost captura de forma más efectiva las interacciones no lineales y 
patrones complejos del dataset, manteniendo una excelente capacidad de generalización, lo que lo convierte en la 
opción más sólida para continuar con la optimización y el pipeline productivo.

"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score
)

import xgboost as xgb

from src.features.feat_eng_pipeline import FeatureEngineeringPipeline


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Logging
# =========================================================
LOG_PATH = ARTIFACTS_DIR / "xgb_optuna_cv.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# =========================================================
# Config
# =========================================================
TARGET_COL = "target_bin"
N_SPLITS = 4
N_TRIALS = 20
RANDOM_STATE = 42
MIN_ACCURACY = 0.86


# =========================================================
# Feature Engineering config
# =========================================================
cat_cols = ['seller_address_country_name', 'seller_address_country_id',
       'seller_address_state_name', 'seller_address_state_id',
       'seller_address_city_name', 'seller_address_city_id', 'warranty',
       'sub_status', 'deal_ids', 'shipping_methods', 'shipping_tags',
       'shipping_mode', 'shipping_dimensions',
       'non_mercado_pago_payment_methods', 'variations', 'site_id',
       'listing_type_id', 'attributes', 'buying_mode', 'tags',
       'listing_source', 'parent_item_id', 'coverage_areas', 'category_id',
       'descriptions', 'last_updated', 'international_delivery_mode',
       'pictures', 'id', 'differential_pricing', 'currency_id', 'thumbnail',
       'title', 'date_created', 'secure_thumbnail', 'status', 'video_id',
       'subtitle', 'permalink', 'shipping_free_methods', 'seller_id'] #Las categoricas que vienen en el df inicial

ts_cols = ["start_time", "stop_time"] #Cols a formatear como datetime

null_tokens = {'', ' ', 'na', 'n/a', 'none', 'null', 'nan', '[]', '{}'}

keep_cols = ['seller_address_state_id', 'shipping_local_pick_up',
       'shipping_free_shipping', 'shipping_mode', 'buying_mode',
       'accepts_mercadopago', 'automatic_relist', 'status',
       'warranty_clean',
       'non_mp_payment_clean', 'listing_type_ord', 'has_good_tag',
       'has_poor_tag', 'tags_count', 'has_description', 'title_cluster',
       'n_images', 'max_size_pixels', 'day_of_week', 'hour', 'day_of_month',
       'month', 'is_weekend', 'is_business_hour', 'part_of_day', 'target_bin',
       'price_log', 'sold_quantity_log', 'initial_quantity_log',
       'max_size_pixels_log', 'sell_through_rate', 'sold_ratio_log',
       'image_density']

bool_cols = ['shipping_local_pick_up', 'shipping_free_shipping', 'accepts_mercadopago', 'automatic_relist', 
             'has_good_tag','has_poor_tag', 'has_description', 'is_weekend', 'is_business_hour'] 
ohe_cols = ['shipping_mode', 'buying_mode', 'status']
freq_enc_cols = ['part_of_day']
target_enc_cols = ['title_cluster', 'non_mp_payment_clean', 'warranty_clean', 'seller_address_state_id']

# =========================================================
# Metrics helper
# =========================================================
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "recall_used": recall_score(y_true, y_pred, pos_label=1),
        "precision_used": precision_score(y_true, y_pred, pos_label=1)
    }


# =========================================================
# Optuna objective
# =========================================================
def objective(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",  
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):

        X_tr_raw, X_val_raw = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        fe = FeatureEngineeringPipeline(
            categorical_cols=cat_cols,
            ts_cols=ts_cols,
            null_tokens=null_tokens,
            columns_to_keep=keep_cols,
            bool_cols=bool_cols,
            ohe_cols=ohe_cols,
            target_enc_cols=target_enc_cols,
            freq_enc_cols=freq_enc_cols
        )

        fe.fit(X_tr_raw, y_tr)
        X_tr = fe.transform(X_tr_raw)
        X_val = fe.transform(X_val_raw)

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_val_pred, y_val_proba)
        fold_metrics.append(metrics)

    df_folds = pd.DataFrame(fold_metrics)
    mean_metrics = df_folds.mean()

    # ---- Hard constraint: accuracy
    if mean_metrics["accuracy"] < MIN_ACCURACY:
        raise optuna.exceptions.TrialPruned()

    # Store metrics
    trial.set_user_attr("accuracy", mean_metrics["accuracy"])
    trial.set_user_attr("recall_used", mean_metrics["recall_used"])
    trial.set_user_attr("precision_used", mean_metrics["precision_used"])

    return mean_metrics["roc_auc"]


# =========================================================
# Main
# =========================================================
def main():

    logger.info("🚀 Starting XGBoost Optuna CV optimization")

    df = pd.read_parquet(DATA_DIR / "train_base.parquet")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Used ratio: {y.mean():.4f}")

    study = optuna.create_study(
        direction="maximize",
        study_name="xgb_optuna_cv"
    )

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    logger.info("✅ Optimization finished")
    logger.info(f"Best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    records = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            records.append({
                **t.params,
                "val_auc": t.value,
                "val_accuracy": t.user_attrs["accuracy"],
                "val_recall_used": t.user_attrs["recall_used"],
                "val_precision_used": t.user_attrs["precision_used"]
            })

    results_df = pd.DataFrame(records)
    out_path = ARTIFACTS_DIR / "xgb_optuna_cv_results.csv"
    results_df.to_csv(out_path, index=False)

    logger.info(f"📊 Results saved to {out_path}")
    logger.info("🏁 XGBoost Optuna experiments finished")


if __name__ == "__main__":
    main()
