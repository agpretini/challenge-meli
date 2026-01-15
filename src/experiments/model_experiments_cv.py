"""Model experiments with cross-validation.
En este script se evalúan distintos algoritmos de clasificación supervisada (Random Forest, LightGBM, XGBoost y CatBoost) 
sobre el conjunto de entrenamiento base, utilizando validación cruzada estratificada.
El objetivo comparar modelos de forma justa y reproducible, evitando explícitamente cualquier tipo de data leakage. Para ello, en cada fold:

- El Feature Engineering se fitea únicamente sobre el subset de entrenamiento.
- El subset de validación se transforma usando solo los parámetros aprendidos en train.
- Las métricas se calculan de manera independiente en train y validación.

La métrica principal de evaluación es Accuracy, con un objetivo mínimo de 0.86, 
Accuracy se utiliza como métrica principal por requerimiento del challenge.
ROC-AUC se emplea como métrica secundaria para comparar capacidad discriminativa entre modelos, 
mientras que el recall de la clase used se monitorea para controlar que no haya un gran impacto de falsos negativos 
sobre la clase de mayor interés (seria un problema no detectar un producto que es usado como tal, podria arruinar la experiencia
de muchos usuarios).
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score
)

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from src.features.feat_eng_pipeline import FeatureEngineeringPipeline


# ===============================
# Paths
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# Logging
# ===============================
LOG_PATH = ARTIFACTS_DIR / "model_experiments_cv.log"

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
N_SPLITS = 5
RANDOM_STATE = 42

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

# ===============================
# Models
# ===============================
def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_depth=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=10,

            # Regularization
            l2_leaf_reg=2.0,
            min_data_in_leaf=20,

            loss_function="Logloss",
            eval_metric="AUC",

            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            # Tree structure
            max_depth=10,
            num_leaves=31,              # ≈ 2^5 → complejidad moderada

            # Sampling
            subsample=0.8,
            colsample_bytree=0.8,
            subsample_freq=1,

            # Regularization (suave)
            min_child_samples=20,
            reg_alpha=0.0,
            reg_lambda=1.0,

            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            
            # Tree structure
            max_depth=10,
            min_child_weight=1,         # ≈ min_child_samples en LGBM (controlan la complejidad de los splits, generando un efecto similar)

            # Sampling
            subsample=0.8,
            colsample_bytree=0.8,

            # Regularization (match LGBM)
            reg_alpha=0.0,
            reg_lambda=1.0,

            objective="binary:logistic",
            eval_metric="logloss",

            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


# ===============================
# Metrics
# ===============================
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "recall_used": recall_score(y_true, y_pred, pos_label=1),
        "precision_used": precision_score(y_true, y_pred, pos_label=1)
    }


# ===============================
# Main CV Loop
# ===============================
def main():
    logger.info("Starting model experiments")

    df = pd.read_parquet(DATA_DIR / "train_base.parquet")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Used ratio: {y.mean():.4f}")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_results = []

    for model_name, model in get_models().items():
        logger.info(f"🧪 Model: {model_name}")

        fold_results = []

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
            logger.info(f"📂 Fold {fold}")

            X_tr_raw, X_val_raw = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            # -------- FE (fit SOLO en train)
            fe = FeatureEngineeringPipeline(
                categorical_cols = cat_cols,
                ts_cols = ts_cols,
                null_tokens = null_tokens,
                columns_to_keep = keep_cols,
                bool_cols = bool_cols,
                ohe_cols = ohe_cols,
                target_enc_cols = target_enc_cols,
                freq_enc_cols = freq_enc_cols
            )

            fe.fit(X_tr_raw, y_tr)

            X_tr = fe.transform(X_tr_raw)
            X_val = fe.transform(X_val_raw)

            # -------- Model
            model.fit(X_tr, y_tr)

            # -------- Train metrics
            y_tr_pred = model.predict(X_tr)
            y_tr_proba = model.predict_proba(X_tr)[:, 1]
            train_metrics = compute_metrics(y_tr, y_tr_pred, y_tr_proba)

            # -------- Val metrics
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba)

            logger.info(
                f"Fold {fold} | "
                f"Train Acc={train_metrics['accuracy']:.4f} | "
                f"Val Acc={val_metrics['accuracy']:.4f} | "
                f"Val Recall={val_metrics['recall_used']:.4f}"
            )

            fold_results.append({
                "model": model_name,
                "fold": fold,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

        df_folds = pd.DataFrame(fold_results)
        summary = df_folds.mean(numeric_only=True).to_dict()
        summary["model"] = model_name
        summary["fold"] = "mean"

        all_results.extend(fold_results)
        all_results.append(summary)

        logger.info(
            f"✅ {model_name} CV Mean | "
            f"Val Acc={summary['val_accuracy']:.4f} | "
            f"Val AUC={summary['val_roc_auc']:.4f} | "
            f"Val Recall={summary['val_recall_used']:.4f}"
        )

    results_df = pd.DataFrame(all_results)
    out_path = ARTIFACTS_DIR / "model_experiments_cv_results.csv"
    results_df.to_csv(out_path, index=False)

    logger.info(f"📊 Results saved to {out_path}")
    logger.info("🏁 Experiments finished")


if __name__ == "__main__":
    main()
