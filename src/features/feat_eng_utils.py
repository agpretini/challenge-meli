""" Este modulo define una clase para cada transformacion de feature engineering a realizar en el pipeline. 
Se decidió implementar el feature engineering mediante transformadores con interfaz fit/transform, 
luego centralizados en un único módulo (FE_utils.py) para mantener simplicidad y reutilización.
Un pipeline orquestador (FE_pipeline.py) permite aplicar exactamente las mismas transformaciones sobre train, 
validation y test, evitando data leakage y asegurando consistencia """

##===== IMPORTS =====##
import pandas as pd
import numpy as np
import math
from pandas.api.types import is_scalar
import unicodedata
import re
import ast
from typing import Iterable, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

##===== NULL NORMALIZER =====##
class CategoricalNullNormalizer:
    """
    Normaliza representaciones de nulos en columnas categóricas.
    Transformer sin estado.
    """

    def __init__(
        self,
        columns: list[str],
        null_tokens: set[str] | None = None
    ):
        self.columns = columns
        self.null_tokens = (
            null_tokens
            if null_tokens is not None
            else {'', ' ', 'na', 'n/a', 'none', 'null', 'nan', '[]', '{}'}
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].map(self._to_na)
        return X

    def _to_na(self, x):
        if isinstance(x, np.ndarray):
            return pd.NA if x.size == 0 else str(x)

        if isinstance(x, (list, tuple, set, dict)):
            return pd.NA if len(x) == 0 else str(x)

        if x is None or pd.isna(x):
            return pd.NA

        x_str = str(x).strip().lower()
        if x_str in self.null_tokens:
            return pd.NA

        return x

##===== LOW VARIANCE FILTER =====##
class LowVarianceFilter:
    """
    Detecta y elimina columnas con:
        - varianza nula
        - varianza casi nula
        - completamente nulas

    La detección se hace SOLO en fit (train).
    """

    def __init__(
        self,
        freq_threshold: float = 0.99,
        min_unique: int = 2
    ):
        self.freq_threshold = freq_threshold
        self.min_unique = min_unique

        self.low_variance_df_: pd.DataFrame | None = None
        self.columns_to_drop_: list[str] = []

    @staticmethod
    def _to_hashable(v):
        if isinstance(v, (list, dict, np.ndarray)):
            return str(v)

        if v is None:
            return pd.NA

        if isinstance(v, float) and math.isnan(v):
            return pd.NA

        try:
            if pd.isna(v):
                return pd.NA
        except Exception:
            pass

        if not is_scalar(v):
            return str(v)

        return v

    def fit(self, X: pd.DataFrame, y=None):
        records = []

        for col in X.columns:
            s = X[col].map(self._to_hashable)

            # columna completamente nula
            if s.notna().sum() == 0:
                records.append({
                    "column": col,
                    "type": "all_null",
                    "n_unique": 0,
                    "top_freq": 1.0
                })
                continue

            n_unique = s.nunique(dropna=True)
            vc = s.value_counts(dropna=True, normalize=True)
            top_freq = vc.iloc[0]

            if n_unique < self.min_unique:
                records.append({
                    "column": col,
                    "type": "zero_variance",
                    "n_unique": n_unique,
                    "top_freq": top_freq
                })
            elif top_freq >= self.freq_threshold:
                records.append({
                    "column": col,
                    "type": "near_zero_variance",
                    "n_unique": n_unique,
                    "top_freq": top_freq
                })

        self.low_variance_df_ = (
            pd.DataFrame(records)
            .sort_values(["type", "top_freq"], ascending=[True, False])
            .reset_index(drop=True)
        )

        self.columns_to_drop_ = self.low_variance_df_["column"].tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_drop_ is None:
            raise RuntimeError("LowVarianceFilter must be fitted first")

        return X.drop(columns=self.columns_to_drop_, errors="ignore")
    

##===== HIGH NULL FILTER =====##
class HighNullFilter:
    """
    Elimina columnas con una proporción de nulos mayor o igual a un threshold.
    La decisión se toma SOLO en fit (train).
    """

    def __init__(self, null_threshold: float = 0.8):
        self.null_threshold = null_threshold
        self.columns_to_drop_: list[str] = []
        self.null_ratio_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y=None):
        # Proporción de nulos por columna
        self.null_ratio_ = X.isna().mean()

        self.columns_to_drop_ = (
            self.null_ratio_[self.null_ratio_ >= self.null_threshold]
            .index
            .tolist()
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_drop_ is None:
            raise RuntimeError("HighNullFilter must be fitted first")

        return X.drop(columns=self.columns_to_drop_, errors="ignore")
    

##==== = COLUMN TYPE NORMALIZER =====##
class ColumnTypeNormalizer:
    """
    Normaliza tipos de columnas:
    - Castea columnas categóricas
    - Convierte columnas timestamp a datetime
    No requiere fit (transformación determinística).
    """

    def __init__(
        self,
        categorical_cols: list[str] | None = None,
        timestamp_cols: list[str] | None = None,
        timestamp_unit: str = "ms"
    ):
        self.categorical_cols = categorical_cols or []
        self.timestamp_cols = timestamp_cols or []
        self.timestamp_unit = timestamp_unit

    def fit(self, X: pd.DataFrame, y=None):
        # No aprende nada
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # -------- categóricas --------
        for col in self.categorical_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype("category")

        # -------- timestamps --------
        for col in self.timestamp_cols:
            if col in X_out.columns:
                X_out[col] = pd.to_datetime(
                    X_out[col],
                    unit=self.timestamp_unit,
                    errors="coerce"
                )

        return X_out

##==== = HIGH CORRELATION FILTER =====##
class HighCorrelationFilter:
    """
    Elimina features numéricas altamente correlacionadas usando
    correlación de Spearman. Decisiones se toman SOLO en fit().
    """

    def __init__(
        self,
        threshold: float = 0.99,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.threshold = threshold
        self.random_state = random_state
        self.verbose = verbose

        self.columns_to_drop_: list[str] = []
        self.corr_summary_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y=None):
        rng = np.random.default_rng(self.random_state)

        num_cols = X.select_dtypes(include="number").columns.tolist()

        if len(num_cols) < 2:
            if self.verbose:
                print("⚠️ Not enough numerical columns for correlation analysis.")
            self.columns_to_drop_ = []
            self.corr_summary_ = pd.DataFrame()
            return self

        corr_matrix = X[num_cols].corr(method="spearman")

        records = []
        to_drop = set()
        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = corr_matrix.iloc[i, j]

                if abs(corr) >= self.threshold:
                    f1, f2 = cols[i], cols[j]

                    if f1 in to_drop or f2 in to_drop:
                        continue

                    nulls_f1 = X[f1].isna().mean()
                    nulls_f2 = X[f2].isna().mean()

                    if nulls_f1 > nulls_f2:
                        drop, keep = f1, f2
                        decision = "more_nulls"
                    elif nulls_f2 > nulls_f1:
                        drop, keep = f2, f1
                        decision = "more_nulls"
                    else:
                        drop = rng.choice([f1, f2])
                        keep = f2 if drop == f1 else f1
                        decision = "random_same_nulls"

                    to_drop.add(drop)

                    records.append({
                        "feature_1": f1,
                        "feature_2": f2,
                        "spearman_corr": corr,
                        "kept_feature": keep,
                        "dropped_feature": drop,
                        "decision_rule": decision,
                        "null_rate_f1": round(nulls_f1, 4),
                        "null_rate_f2": round(nulls_f2, 4)
                    })

        summary_df = pd.DataFrame(records)

        if not summary_df.empty:
            summary_df = summary_df.sort_values(
                "spearman_corr",
                key=lambda s: s.abs(),
                ascending=False
            )

        self.columns_to_drop_ = sorted(to_drop)
        self.corr_summary_ = summary_df

        if self.verbose:
            print(f"\n📊 High Correlation Summary (Spearman ≥ {self.threshold})")
            print("-" * 80)

            if summary_df.empty:
                print("✅ No highly correlated numerical features detected.")
            else:
                for _, row in summary_df.iterrows():
                    print(
                        f"{row['feature_1']} <-> {row['feature_2']} | "
                        f"ρ={row['spearman_corr']:.3f} | "
                        f"DROP: {row['dropped_feature']} | "
                        f"KEEP: {row['kept_feature']} | "
                        f"RULE: {row['decision_rule']}"
                    )

            print("\n🗑️ Columns dropped:", self.columns_to_drop_)
            print(f"📉 Total dropped: {len(self.columns_to_drop_)}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.columns_to_drop_:
            return X

        return X.drop(columns=self.columns_to_drop_, errors="ignore")
    

##==== = WARRANTY CLEANER =====##
class WarrantyCleaner:
    """
    Limpia y normaliza la columna warranty.
    Las categorías frecuentes se aprenden SOLO en fit().
    """

    def __init__(
        self,
        min_freq: int = 100,
        other_label: str = "other",
        null_label: str = "sin_garantia"
    ):
        self.min_freq = min_freq
        self.other_label = other_label
        self.null_label = null_label

        self.frequent_values_: set[str] | None = None

    @staticmethod
    def _normalize_text(x: str) -> str:
        x = unicodedata.normalize("NFKD", x)
        x = x.encode("ascii", "ignore").decode("utf-8")
        x = re.sub(r"[^a-zA-Z0-9\s]", " ", x)
        x = re.sub(r"\s+", " ", x).strip().lower()
        return x

    def _semantic_normalization(self, s: pd.Series) -> pd.Series:
        return s.replace(
            {
                # ---------- años -> meses ----------
                r".*\b(1|un)\s+ano\b.*": "12 meses",
                r".*\bano\s+de\s+garantia\b.*": "12 meses",

                # ---------- meses ----------
                r".*\b6\s+meses\b.*": "6 meses",
                r".*\b3\s+meses\b.*": "3 meses",

                # ---------- dias ----------
                r".*\b60\s+dias\b.*": "2 meses",

                # ---------- garantia explicita ----------
                r".*con\s+garantia.*": "si",
                r".*100\s+garantizados.*": "si",
                r".*garantia\s+total.*": "si",
                r".*se\s+garantiza\s+la\s+descripcion.*": "si",
                r".*defectos\s+de\s+fabricacion.*": "si",

                # ---------- fabricante ----------
                r".*de\s+fabrica.*": "si",
                r".*del\s+fabricante.*": "si",
                r".*garantia\s+de\s+fabrica.*": "si",

                # ---------- reputacional ----------
                r".*mi\s+reputacion.*": "garantia_reputacional",
                r".*mis\s+calificaciones.*": "garantia_reputacional",
                r".*revisa\s+nuestras\s+calificaciones.*": "garantia_reputacional",
                r".*oferta\s+con\s+confianza.*": "garantia_reputacional",

                # ---------- textos logisticos ----------
                r".*caja\s+sellada.*": self.other_label,
            },
            regex=True
        )

    def fit(self, series: pd.Series, y=None):
        s = series.copy()

        # 1) imputar nulos
        s = s.fillna(self.null_label)

        # 2) normalización base
        s = s.astype(str).map(self._normalize_text)

        # 3) normalización semántica
        s = self._semantic_normalization(s)

        # 4) aprender categorías frecuentes
        vc = s.value_counts()
        self.frequent_values_ = set(vc[vc >= self.min_freq].index)

        return self

    def transform(self, series: pd.Series) -> pd.Series:
        if self.frequent_values_ is None:
            raise RuntimeError("WarrantyCleaner must be fitted first")

        s = series.copy()

        s = s.fillna(self.null_label)
        s = s.astype(str).map(self._normalize_text)
        s = self._semantic_normalization(s)

        s = s.where(s.isin(self.frequent_values_), self.other_label)

        return s


##==== = NON MP PAYMENT CLEANER =====##
import pandas as pd
import re
from typing import Iterable


class NonMPPaymentCleaner:
    """
    Limpia y resume métodos de pago no MercadoPago.
    Aprende categorías frecuentes SOLO en train.
    """

    ID_PATTERN = re.compile(r"'id'\s*:\s*'([^']+)'")

    def __init__(
        self,
        min_freq: int = 100,
        other_label: str = "other"
    ):
        self.min_freq = min_freq
        self.other_label = other_label
        self.frequent_categories_: set[str] | None = None

    # -------------------------------------------------
    # Parsing
    # -------------------------------------------------
    @classmethod
    def extract_payment_ids(cls, x) -> list[str]:
        if x is None or pd.isna(x):
            return []
        if isinstance(x, str):
            return cls.ID_PATTERN.findall(x)
        if isinstance(x, (list, tuple)):
            return list(x)
        return []

    @staticmethod
    def _summarize(parsed: Iterable[str]) -> str:
        if len(parsed) == 0:
            return "solo_mp"

        accepts_cash = "MLAMO" in parsed
        accepts_transfer = "MLATB" in parsed
        accepts_agreement = "MLAWC" in parsed
        accepts_card = any(
            i.startswith("MLA") and i not in {"MLAMO", "MLATB", "MLAWC"}
            for i in parsed
        )

        keys = []
        if accepts_cash:
            keys.append("efectivo")
        if accepts_transfer:
            keys.append("transferencia")
        if accepts_card:
            keys.append("tarjeta")
        if accepts_agreement:
            keys.append("acordar")

        if len(keys) == 1:
            return keys[0]
        if len(keys) <= 3:
            return "_".join(sorted(keys))
        return "mixto"

    # -------------------------------------------------
    # API pipeline
    # -------------------------------------------------
    def fit(self, series: pd.Series):
        parsed = series.map(self.extract_payment_ids)
        summary = parsed.map(self._summarize)

        vc = summary.value_counts()
        self.frequent_categories_ = set(
            vc[vc >= self.min_freq].index
        )

        return self

    def transform(self, series: pd.Series) -> pd.Series:
        if self.frequent_categories_ is None:
            raise RuntimeError(
                "NonMPPaymentCleaner must be fitted before transform"
            )

        parsed = series.map(self.extract_payment_ids)
        summary = parsed.map(self._summarize)

        return summary.where(
            summary.isin(self.frequent_categories_),
            self.other_label
        )


##==== = LISTING TYPE ORDINAL ENCODER =====##
class ListingTypeOrdinalEncoder:
    """
    Codifica listing_type_id como variable ordinal.
    Orden fijo de negocio (peor -> mejor).
    """

    DEFAULT_MAPPING = {
        "free": 1,
        "bronze": 2,
        "silver": 3,
        "gold": 4,
        "gold_special": 5,
        "gold_premium": 6,
        "gold_pro": 7,
    }

    def __init__(
        self,
        mapping: dict[str, int] | None = None,
        unknown_value: int = 0
    ):
        self.mapping = mapping or self.DEFAULT_MAPPING
        self.unknown_value = unknown_value

    def fit(self, series: pd.Series):
        # No aprende nada: mapping fijo
        return self

    def transform(self, series: pd.Series) -> pd.Series:
        return (
            series
            .astype(str)
            .str.lower()
            .map(self.mapping)
            .fillna(self.unknown_value)
            .astype(int)
        )

##==== TAGS FEATURE EXTRACTOR =====##

class TagsFeatureExtractor:
    """
    Procesa una columna 'tags' y genera:
    - has_good_tag
    - has_poor_tag
    - tags_count
    """

    TAG_PATTERN = re.compile(r"'([^']+)'")

    def __init__(
        self,
        source_col: str = "tags",
        prefix: str = ""
    ):
        self.source_col = source_col
        self.prefix = prefix

    def fit(self, series: pd.Series):
        # Stateless
        return self

    def _parse_tags(self, x):
        if x is None or pd.isna(x):
            return []

        if isinstance(x, list):
            return x

        if isinstance(x, str):
            x = x.strip()
            if x in ("", "[]", "nan"):
                return []
            return self.TAG_PATTERN.findall(x)

        return []

    def transform(self, series: pd.Series) -> pd.DataFrame:
        parsed = series.map(self._parse_tags)

        return pd.DataFrame({
            f"{self.prefix}has_good_tag": parsed.map(
                lambda tags: any("good" in t.lower() for t in tags)
            ).astype(int),

            f"{self.prefix}has_poor_tag": parsed.map(
                lambda tags: any("poor" in t.lower() for t in tags)
            ).astype(int),

            f"{self.prefix}tags_count": parsed.map(len)
        })

##==== DESCRIPTION FLAG ADDER =====##
class DescriptionFlagAdder:
    """
    Agrega un flag binario indicando si el item tiene descripción no vacía.
    """

    def __init__(
        self,
        desc_col: str = "descriptions",
        new_col: str = "has_description"
    ):
        self.desc_col = desc_col
        self.new_col = new_col

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        if self.desc_col not in X_out.columns:
            return X_out

        X_out[self.new_col] = (
            X_out[self.desc_col]
            .astype(str)
            .str.strip()
            .ne("")
            .astype(int)
        )

        # Forzar 0 en NaN originales
        X_out.loc[X_out[self.desc_col].isna(), self.new_col] = 0

        return X_out
    

##==== TITLE CLUSTER ENCODER =====##
class TitleClusterEncoder:
    """
    Genera clusters de títulos usando:
    TF-IDF -> TruncatedSVD -> KMeans

    Output:
    - una columna categórica: title_cluster
    """

    def __init__(
        self,
        title_col: str = "title",
        output_col: str = "title_cluster",
        max_features: int = 10_000,
        min_df: int = 5,
        ngram_range: tuple = (1, 2),
        svd_components: int = 200,
        n_clusters: int = 25,
        random_state: int = 42
    ):
        self.title_col = title_col
        self.output_col = output_col

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            ngram_range=ngram_range,
            strip_accents="unicode"
        )

        self.svd = TruncatedSVD(
            n_components=svd_components,
            random_state=random_state
        )

        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto"
        )

        self._is_fitted = False

    def _preprocess_titles(self, series: pd.Series) -> pd.Series:
        return (
            series
            .fillna("")
            .astype(str)
            .str.lower()
        )

    def fit(self, X: pd.DataFrame, y=None):
        titles = self._preprocess_titles(X[self.title_col])

        X_tfidf = self.vectorizer.fit_transform(titles)
        X_svd = self.svd.fit_transform(X_tfidf)

        self.kmeans.fit(X_svd)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("TitleClusterEncoder must be fitted first")

        X_out = X.copy()

        titles = self._preprocess_titles(X_out[self.title_col])

        X_tfidf = self.vectorizer.transform(titles)
        X_svd = self.svd.transform(X_tfidf)

        clusters = self.kmeans.predict(X_svd)

        X_out[self.output_col] = clusters.astype("object")

        return X_out

##==== IMAGE FEATURES EXTRACTOR =====##
class ImageFeaturesExtractor:
    """
    Extrae features simples a partir de una columna de imágenes:
    - n_images: cantidad real de imágenes
    - max_size_pixels: máxima resolución (w * h)
    """

    def __init__(
        self,
        image_col: str = "pictures",
        n_images_col: str = "n_images",
        max_size_col: str = "max_size_pixels"
    ):
        self.image_col = image_col
        self.n_images_col = n_images_col
        self.max_size_col = max_size_col

    def fit(self, X, y=None):
        # No hay nada que aprender
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        n_images = []
        max_size_pixels = []

        for val in X_out[self.image_col]:
            imgs = []

            # Caso lista real
            if isinstance(val, list):
                imgs = val

            # Caso string
            elif isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        imgs = parsed
                except Exception:
                    # Fallback robusto
                    n_imgs = len(re.findall(r"\{'id':", val))
                    n_images.append(n_imgs)
                    max_size_pixels.append(0)
                    continue

            # Conteo estándar
            n_images.append(len(imgs))

            max_pix = 0
            for img in imgs:
                if isinstance(img, dict):
                    size_str = img.get("max_size", "")
                    if "x" in size_str:
                        try:
                            w, h = map(int, size_str.split("x"))
                            max_pix = max(max_pix, w * h)
                        except Exception:
                            pass

            max_size_pixels.append(max_pix)

        X_out[self.n_images_col] = n_images
        X_out[self.max_size_col] = max_size_pixels

        return X_out

##==== TIME FEATURES EXTRACTOR =====##
class TimeFeaturesExtractor:
    """
    Genera features temporales seguras (sin leakage) a partir de una columna datetime.
    """

    def __init__(
        self,
        date_col: str = "start_time",
        prefix: str | None = None
    ):
        """
        Parameters
        ----------
        date_col : str
            Columna datetime origen
        prefix : str | None
            Prefijo opcional para las features generadas
        """
        self.date_col = date_col
        self.prefix = prefix

    def fit(self, X, y=None):
        # No hay estado que aprender
        return self

    def _col(self, name: str) -> str:
        return f"{self.prefix}_{name}" if self.prefix else name

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # Asegurar datetime
        dt = pd.to_datetime(X_out[self.date_col], errors="coerce")

        # Básicas
        X_out[self._col("day_of_week")] = dt.dt.dayofweek
        X_out[self._col("hour")] = dt.dt.hour
        X_out[self._col("day_of_month")] = dt.dt.day
        X_out[self._col("month")] = dt.dt.month

        # Flags
        X_out[self._col("is_weekend")] = (
            X_out[self._col("day_of_week")].isin([5, 6])).astype(int)

        X_out[self._col("is_business_hour")] = (
            (X_out[self._col("hour")] >= 9) &
            (X_out[self._col("hour")] <= 18) &
            (~X_out[self._col("is_weekend")])).astype(int)

        # Parte del día
        X_out[self._col("part_of_day")] = pd.cut(
            X_out[self._col("hour")],
            bins=[-1, 5, 11, 17, 23],
            labels=["night", "morning", "afternoon", "evening"])

        return X_out


##==== NUMERIC FEATURES EXTRACTOR =====##
class NumericFeaturesGenerator:
    """
    Genera features numéricas derivadas de forma segura (sin leakage).
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # No aprende parámetros
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        out = pd.DataFrame(index=X_out.index)

        # ------------------------
        # Log-transformations
        # ------------------------
        if "price" in X_out.columns:
            out["price_log"] = np.log1p(X_out["price"])

        if "sold_quantity" in X_out.columns:
            out["sold_quantity_log"] = np.log1p(X_out["sold_quantity"])

        if "initial_quantity" in X_out.columns:
            out["initial_quantity_log"] = np.log1p(X_out["initial_quantity"])

        if "max_size_pixels" in X_out.columns:
            out["max_size_pixels_log"] = np.log1p(X_out["max_size_pixels"])

        # ------------------------
        # Demand-related features
        # ------------------------
        if {"sold_quantity", "initial_quantity"}.issubset(X_out.columns):
            out["sell_through_rate"] = (
                X_out["sold_quantity"] /
                X_out["initial_quantity"].replace(0, np.nan)
            )

            if (
                "sold_quantity_log" in out.columns and
                "initial_quantity_log" in out.columns
            ):
                out["sold_ratio_log"] = (
                    out["sold_quantity_log"] /
                    out["initial_quantity_log"].replace(0, np.nan)
                )

        # ------------------------
        # Image-related feature
        # ------------------------
        if {"max_size_pixels", "n_images"}.issubset(X_out.columns):
            out["image_density"] = (
                X_out["max_size_pixels"] /
                X_out["n_images"].replace(0, np.nan)
            )

        # Concatenamos sin pisar columnas existentes
        return pd.concat([X_out, out], axis=1)

##==== COLUMN SELECTOR =====##
class ColumnSelector:
    """
    Selecciona un subconjunto de columnas al final del pipeline.
    """

    def __init__(
        self,
        columns_to_keep: Optional[Iterable[str]] = None,
        columns_to_drop: Optional[Iterable[str]] = None
    ):
        if columns_to_keep is None and columns_to_drop is None:
            raise ValueError(
                "You must provide either columns_to_keep or columns_to_drop"
            )

        self.columns_to_keep = list(columns_to_keep) if columns_to_keep else None
        self.columns_to_drop = list(columns_to_drop) if columns_to_drop else None

    def fit(self, X, y=None):
        # No aprende parámetros
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        if self.columns_to_keep is not None:
            cols = [c for c in self.columns_to_keep if c in X_out.columns]
            return X_out[cols]

        if self.columns_to_drop is not None:
            cols = [c for c in self.columns_to_drop if c in X_out.columns]
            return X_out.drop(columns=cols, errors="ignore")

        return X_out

###==== MISSING VALUE IMPUTER =====##
class MissingValueImputer:
    """
    Imputa valores faltantes:
    - Numéricas → mediana
    - Categóricas / object / bool → moda
    """

    def __init__(self):
        self.numeric_medians = {}
        self.categorical_modes = {}
        self._is_fitted = False

    def fit(self, X: pd.DataFrame):
        X = X.copy()

        # Numéricas
        num_cols = X.select_dtypes(include="number").columns
        for col in num_cols:
            self.numeric_medians[col] = X[col].median()

        # Categóricas + booleanas
        cat_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        for col in cat_cols:
            if X[col].dropna().empty:
                self.categorical_modes[col] = pd.NA
            else:
                self.categorical_modes[col] = X[col].mode(dropna=True).iloc[0]

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("MissingValueImputer must be fitted first")

        X_out = X.copy()

        # Numéricas
        for col, median in self.numeric_medians.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(median)

        # Categóricas / booleanas
        for col, mode in self.categorical_modes.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(mode)

        return X_out

##====BOOL NORMALIZER   ====
class BooleanNormalizer:
    """
    Normaliza columnas booleanas a int {0,1}.
    Acepta bool, int, float, object con True/False.
    """

    def __init__(self, bool_cols: list[str]):
        self.bool_cols = bool_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in self.bool_cols:
            if col not in X.columns:
                continue

            # Cast seguro
            X[col] = (
                X[col]
                .astype("boolean", errors="ignore")
                .map({True: 1, False: 0})
                .astype("Int64")  # nullable int
            )

        return X

##====OHE Wrapper====
class OneHotEncoderWrapper:

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop=None
        )
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y=None):
        self.encoder.fit(X[self.columns])
        self.feature_names_ = self.encoder.get_feature_names_out(self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = self.encoder.transform(X[self.columns])
        df_encoded = pd.DataFrame(
            encoded,
            columns=self.feature_names_,
            index=X.index
        )
        return df_encoded

 ##====FREQUENCY ENCODER====   
class FrequencyEncoder:

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.freq_maps_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        for col in self.columns:
            freq = X[col].value_counts(normalize=True)
            self.freq_maps_[col] = freq
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)

        for col in self.columns:
            out[f"{col}_freq"] = (
                X[col]
                .map(self.freq_maps_[col])
                .astype(float)
                .fillna(0.0) )

        return out

##====TARGET ENCODER====    
class TargetEncoder:

    def __init__(
        self,
        columns: list[str],
        smoothing: float = 10.0
    ):
        self.columns = columns
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_maps_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean_ = y.mean()

        for col in self.columns:
            stats = (
                pd.DataFrame({"cat": X[col], "y": y})
                .groupby("cat")["y"]
                .agg(["mean", "count"]) )

            smooth = (
                (stats["mean"] * stats["count"] +
                 self.global_mean_ * self.smoothing)
                / (stats["count"] + self.smoothing) )

            self.encoding_maps_[col] = smooth

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)

        for col in self.columns:
            out[f"{col}_te"] = (
                X[col]
                .map(self.encoding_maps_[col])
                .fillna(self.global_mean_) )

        return out


