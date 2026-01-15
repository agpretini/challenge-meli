##===== IMPORTS =====##
import pandas as pd
from pathlib import Path
import joblib
import logging
import time

from src.features.feat_eng_utils import (
    ColumnTypeNormalizer,
    CategoricalNullNormalizer,
    HighNullFilter,
    LowVarianceFilter,
    HighCorrelationFilter,
    WarrantyCleaner,
    NonMPPaymentCleaner,
    ListingTypeOrdinalEncoder,
    TagsFeatureExtractor,
    DescriptionFlagAdder,
    TitleClusterEncoder,
    ImageFeaturesExtractor,
    TimeFeaturesExtractor,
    NumericFeaturesGenerator,
    ColumnSelector,
    MissingValueImputer,
    BooleanNormalizer,
    OneHotEncoderWrapper,
    TargetEncoder,
    FrequencyEncoder,
)

##===== CLASE ORQUESTADORA DEL FE =====##
class FeatureEngineeringPipeline:

    def __init__(
        self,
        categorical_cols: list[str],
        ts_cols: list[str],
        null_tokens: set[str] | None = None,
        high_null_threshold: float = 0.8,
        low_var_freq_threshold: float = 0.99,
        low_var_min_unique: int = 2,
        corr_threshold: float = 0.99,
        warranty_col: str = "warranty",
        min_warranty_freq: int = 100,
        non_mp_col: str = "non_mercado_pago_payment_methods",
        min_non_mp_freq: int = 100,
        listing_type_col: str = "listing_type_id",
        listing_type_ord_col: str = "listing_type_ord",
        tags_col: str = "tags",
        desc_col: str = "descriptions",
        has_desc_col: str = "has_description",
        title_col: str = "title",
        enable_title_clusters: bool = True,
        image_col: str = "pictures",
        enable_image_features: bool = True,
        time_col: str = "start_time",
        enable_time_features: bool = True,
        enable_numeric_features: bool = True,
        columns_to_keep: list[str] | None = None,
        columns_to_drop: list[str] | None = None,
        bool_cols: list[str] | None = None,
        ohe_cols: list[str] | None = None,
        target_enc_cols: list[str] | None = None,
        freq_enc_cols: list[str] | None = None,
        target_col: str = "target_bin",
        **kwargs
    ):

        self.type_normalizer = ColumnTypeNormalizer(
            categorical_cols=["seller_id"],
            timestamp_cols=ts_cols
        )

        self.cat_null_normalizer = CategoricalNullNormalizer(
            columns=categorical_cols,
            null_tokens=null_tokens )

        self.high_null_filter = HighNullFilter(
            null_threshold=high_null_threshold )

        self.low_variance_filter = LowVarianceFilter(
            freq_threshold=low_var_freq_threshold,
            min_unique=low_var_min_unique )

        self.high_corr_filter = HighCorrelationFilter(
            threshold=corr_threshold,
            verbose=True )

        self.warranty_col = warranty_col
        self.warranty_cleaner = WarrantyCleaner(
            min_freq=min_warranty_freq )

        self.non_mp_col = non_mp_col
        self.non_mp_cleaner = NonMPPaymentCleaner(
            min_freq=min_non_mp_freq )

        self.listing_type_col = listing_type_col
        self.listing_type_ord_col = listing_type_ord_col

        self.listing_type_encoder = ListingTypeOrdinalEncoder()

        self.tags_col = tags_col

        self.tags_feature_extractor = TagsFeatureExtractor(
            source_col=tags_col )

        self.desc_col = desc_col

        self.description_flag_adder = DescriptionFlagAdder(
            desc_col=desc_col,
            new_col=has_desc_col )

        self.enable_title_clusters = enable_title_clusters

        if enable_title_clusters:
            self.title_cluster_encoder = TitleClusterEncoder(
                title_col=title_col )
        
        self.enable_image_features = enable_image_features

        if enable_image_features:
            self.image_features_extractor = ImageFeaturesExtractor(
                image_col=image_col )

        self.enable_time_features = enable_time_features

        if enable_time_features:
            self.time_features_extractor = TimeFeaturesExtractor(
                date_col=time_col ) 
        
        self.enable_numeric_features = enable_numeric_features

        if enable_numeric_features:
            self.numeric_features_generator = NumericFeaturesGenerator()
        
        self.column_selector = None
        if columns_to_keep or columns_to_drop:
            self.column_selector = ColumnSelector(
                columns_to_keep=columns_to_keep,
                columns_to_drop=columns_to_drop)
        
        self.missing_value_imputer = MissingValueImputer()

        self.bool_cols = bool_cols
        self.ohe_cols = ohe_cols
        self.target_enc_cols = target_enc_cols
        self.freq_enc_cols = freq_enc_cols
        
        self.boolean_normalizer = BooleanNormalizer(self.bool_cols)
        self.ohe_encoder = OneHotEncoderWrapper(self.ohe_cols)
        self.target_encoder = TargetEncoder(self.target_enc_cols)
        self.freq_encoder = FrequencyEncoder(self.freq_enc_cols)
        
        self.target_col = target_col
        
        self._is_fitted = False

    # -------------------------------------------------
    # FIT
    # -------------------------------------------------
    def fit(self, X, y=None):
        X_tmp = X.copy()

        X_tmp = self.type_normalizer.transform(X_tmp)
        X_tmp = self.cat_null_normalizer.fit(X_tmp).transform(X_tmp)

        self.high_null_filter.fit(X_tmp)
        X_tmp = self.high_null_filter.transform(X_tmp)

        self.low_variance_filter.fit(X_tmp)
        X_tmp = self.low_variance_filter.transform(X_tmp)

        self.high_corr_filter.fit(X_tmp)
        X_tmp = self.high_corr_filter.transform(X_tmp)

        if self.warranty_col in X_tmp.columns:
            self.warranty_cleaner.fit(X_tmp[self.warranty_col])
            X_tmp["warranty_clean"] = self.warranty_cleaner.transform(X_tmp[self.warranty_col])

        if self.non_mp_col in X_tmp.columns:
            self.non_mp_cleaner.fit(X_tmp[self.non_mp_col])
            X_tmp["non_mp_payment_clean"] = self.non_mp_cleaner.transform(X_tmp[self.non_mp_col])

        if self.listing_type_col in X_tmp.columns:
            self.listing_type_encoder.fit(X_tmp[self.listing_type_col])
            X_tmp[self.listing_type_ord_col] = self.listing_type_encoder.transform(X_tmp[self.listing_type_col])
        
        if self.tags_col in X_tmp.columns:
            self.tags_feature_extractor.fit(X_tmp[self.tags_col])
            tags_df = self.tags_feature_extractor.transform(X_tmp[self.tags_col])
            X_tmp = pd.concat([X_tmp, tags_df], axis=1)
        
        X_tmp = self.description_flag_adder.fit(X_tmp).transform(X_tmp)
        
        if self.enable_title_clusters:
            self.title_cluster_encoder.fit(X_tmp)
            X_tmp = self.title_cluster_encoder.transform(X_tmp)
        
        if self.enable_image_features:
            self.image_features_extractor.fit(X_tmp)
            X_tmp = self.image_features_extractor.transform(X_tmp)
        
        if self.enable_time_features:
            self.time_features_extractor.fit(X_tmp)
            X_tmp = self.time_features_extractor.transform(X_tmp)
        
        if self.enable_numeric_features:
            self.numeric_features_generator.fit(X_tmp)
            X_tmp = self.numeric_features_generator.transform(X_tmp)

        if self.column_selector:
            self.column_selector.fit(X_tmp)
            X_tmp = self.column_selector.transform(X_tmp)
        
        #---------- IMPUTACION NULOS ------
        self.missing_value_imputer.fit(X_tmp)
        X_tmp = self.missing_value_imputer.transform(X_tmp)
        
        # ---------- ENCODING ----------
        if self.ohe_cols:
            self.ohe_encoder.fit(X_tmp)

        if self.freq_enc_cols:
            self.freq_encoder.fit(X_tmp)

        if self.target_enc_cols:
            self.target_encoder.fit(X_tmp, y)
        
        self._is_fitted = True
        return self

    # -------------------------------------------------
    # TRANSFORM
    # -------------------------------------------------
    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted first")

        X_out = X.copy()

        X_out = self.type_normalizer.transform(X_out)
        X_out = self.cat_null_normalizer.transform(X_out)
        X_out = self.high_null_filter.transform(X_out)
        X_out = self.low_variance_filter.transform(X_out)
        X_out = self.high_corr_filter.transform(X_out)

        if self.warranty_col in X_out.columns:
            X_out["warranty_clean"] = self.warranty_cleaner.transform(
                X_out[self.warranty_col])

        if self.non_mp_col in X_out.columns:
            X_out["non_mp_payment_clean"] = self.non_mp_cleaner.transform(
                X_out[self.non_mp_col])
            
        if self.listing_type_col in X_out.columns:
            X_out[self.listing_type_ord_col] = (
                self.listing_type_encoder.transform(
                    X_out[self.listing_type_col] ))
            
        if self.tags_col in X_out.columns:
            tags_df = self.tags_feature_extractor.transform(
                X_out[self.tags_col])
            X_out = pd.concat([X_out, tags_df], axis=1)

        X_out = self.description_flag_adder.transform(X_out)
        
        if self.enable_title_clusters:
            X_out = self.title_cluster_encoder.transform(X_out)
        
        if self.enable_image_features:
            X_out = self.image_features_extractor.transform(X_out)
        
        if self.enable_time_features:
            X_out = self.time_features_extractor.transform(X_out)

        if self.enable_numeric_features:
            X_out = self.numeric_features_generator.transform(X_out)
        
        if self.column_selector:
            X_out = self.column_selector.transform(X_out)
        
        #-------- IMPUTACION NULOS ------
        X_out = self.missing_value_imputer.transform(X_out)
        
        #------ NORMALIZAR BOOLEANOS ------
        X_out = self.boolean_normalizer.transform(X_out)
        
        #---------- ENCODING ----------
        encoded_parts = []

        if self.ohe_cols:
            encoded_parts.append(
                self.ohe_encoder.transform(X_out) )
            X_out = X_out.drop(columns=self.ohe_cols)

        if self.freq_enc_cols:
            encoded_parts.append(
                self.freq_encoder.transform(X_out) )
            X_out = X_out.drop(columns=self.freq_enc_cols)

        if self.target_enc_cols:
            encoded_parts.append(
                self.target_encoder.transform(X_out) )
            X_out = X_out.drop(columns=self.target_enc_cols)

        if encoded_parts:
            X_out = pd.concat([X_out] + encoded_parts, axis=1)
        
        return X_out
    

##===== EJECUCION =====##
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "processed"
ARTIFACTS_DIR = ROOT_DIR / "data" / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

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

def main():
    start_time = time.time()
    logger.info("🚀 Starting Feature Engineering Pipeline")
    
    # ------------------
    # Load base datasets
    # ------------------
    train = pd.read_parquet(DATA_DIR / "train_base.parquet")
    test  = pd.read_parquet(DATA_DIR / "test_base.parquet")

    logger.info(f"Train base shape: {train.shape}")
    logger.info(f"Test base shape : {test.shape}")
    
    y_train = train["target_bin"]
    y_test = test["target_bin"]
    X_train = train.drop(columns=["target_bin"]).copy()
    X_test  = test.drop(columns=["target_bin"]).copy()

    logger.info(f"X_train shape before FE: {X_train.shape}")
    logger.info(f"X_test shape before FE : {X_test.shape}")
    
    # ------------------
    # Build pipeline
    # ------------------
    fe_pipeline = FeatureEngineeringPipeline(
        categorical_cols = cat_cols,
        ts_cols = ts_cols,
        null_tokens = null_tokens,
        columns_to_keep = keep_cols,
        bool_cols = bool_cols,
        ohe_cols = ohe_cols,
        target_enc_cols = target_enc_cols,
        freq_enc_cols = freq_enc_cols
    )
        
    # ------------------
    # Fit + transform
    # ------------------
    logger.info("🔧 Fitting feature engineering pipeline on train")
    
    X_train_fe = fe_pipeline.fit(X_train, y_train).transform(X_train)
    X_test_fe  = fe_pipeline.transform(X_test)

    # ------------------
    # Post-FE checks
    # ------------------
    logger.info(f"X_train shape after FE: {X_train_fe.shape}")
    logger.info(f"X_test shape after FE : {X_test_fe.shape}")

    # Null checks
    n_nulls_train = X_train_fe.isna().sum().sum()
    n_nulls_test = X_test_fe.isna().sum().sum()

    logger.info(f"Total nulls in train after FE: {n_nulls_train}")
    logger.info(f"Total nulls in test after FE : {n_nulls_test}")

    if n_nulls_train > 0 or n_nulls_test > 0:
        logger.warning("⚠️ There are remaining nulls after feature engineering")

    # ------------------
    # Column consistency
    # ------------------
    train_cols = set(X_train_fe.columns)
    test_cols = set(X_test_fe.columns)

    only_in_train = train_cols - test_cols
    only_in_test = test_cols - train_cols

    logger.info(f"Final number of features: {len(train_cols)}")

    logger.info("📌 Final feature columns:")
    logger.info(sorted(train_cols))

    if only_in_train:
        logger.error(f"Columns only in train: {sorted(only_in_train)}")

    if only_in_test:
        logger.error(f"Columns only in test: {sorted(only_in_test)}")

    if not only_in_train and not only_in_test:
        logger.info("✅ Train and test have identical feature columns")

    
    # ------------------
    # Save outputs
    # ------------------
    train_fe = X_train_fe.copy()
    train_fe["target_bin"] = y_train.values
    test_fe = X_test_fe.copy()
    test_fe["target_bin"] = y_test.values

    train_fe.to_parquet(DATA_DIR / "train_fe.parquet", index=False)
    test_fe.to_parquet(DATA_DIR / "test_fe.parquet", index=False)

    logger.info("💾 Saved processed datasets:")
    logger.info(f" - {DATA_DIR / 'train_fe.parquet'}")
    logger.info(f" - {DATA_DIR / 'test_fe.parquet'}")
    
    joblib.dump(fe_pipeline, ARTIFACTS_DIR / "feat_eng_pipeline.pkl")

    logger.info(f"📦 Saved feature engineering pipeline artifact:")
    logger.info(f" - {ARTIFACTS_DIR / 'feat_eng_pipeline.pkl'}")

    elapsed = time.time() - start_time
    logger.info(f"⏱️ Feature engineering completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

