"""
Script para determinar el punto de corte (threshold) óptimo para clasificación binaria dados los hiperparametros que mejor AUC lograron.
Con la definicion del tresshold se busca maximizar el recall de la clase 'used' (1) manteniendo una accuracy mínima de 0.86.
Metodología:
Usar out-of-fold predictions de CV sobre train.
Pipeline:

1. Usar CV de 4 folds
2. En cada fold:
- Entrenar modelo con FE (fit en train fold)
- Predecir probabilidades sobre validation fold
- Concatenar todas las probabilidades OOF
3. Evaluar distintos thresholds sobre ese set OOF
"""


import numpy as np
import pandas as pd
import logging
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, precision_score

import xgboost as xgb

from src.features.feat_eng_pipeline import FeatureEngineeringPipeline

# ===============================
# Config
# ===============================
TARGET_COL = "target_bin"
N_SPLITS = 4
RANDOM_STATE = 42
MIN_ACCURACY = 0.865 #lo fijo un poquito por encima para tener un poquito de margen ante alguna pequeña variacion que pudiera ocurrir en prodduccion

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "data" / "artifacts"

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
BEST_XGB_PARAMS = { 
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
# Logging
# ===============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# Main
# ===============================
def main():

    df = pd.read_parquet(DATA_DIR / "train_base.parquet")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof_proba = np.zeros(len(X))
    oof_true = y.values

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
        logger.info(f"Fold {fold}")

        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
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
        X_tr_fe = fe.fit(X_tr, y_tr).transform(X_tr)
        X_val_fe = fe.transform(X_val)

        model = xgb.XGBClassifier(
            **BEST_XGB_PARAMS
        )

        model.fit(X_tr_fe, y_tr)
        oof_proba[val_idx] = model.predict_proba(X_val_fe)[:, 1]

    # ===============================
    # Threshold search
    # ===============================
    results = []

    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (oof_proba >= t).astype(int)

        acc = accuracy_score(oof_true, y_pred)
        rec = recall_score(oof_true, y_pred)
        prec = precision_score(oof_true, y_pred)

        if acc >= MIN_ACCURACY:
            results.append({
                "threshold": t,
                "accuracy": acc,
                "recall_used": rec,
                "precision_used": prec
            })

    df_res = pd.DataFrame(results).sort_values(
        ["recall_used", "precision_used"],
        ascending=False
    )

    best = df_res.iloc[0]

    logger.info("✅ Optimal threshold found")
    logger.info(best)

    df_res.to_csv(ARTIFACTS_DIR / "threshold_analysis.csv", index=False)

    with open(ARTIFACTS_DIR / "decision_threshold.txt", "w") as f:
        f.write(str(best["threshold"]))


if __name__ == "__main__":
    main()
