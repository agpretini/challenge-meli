# MercadoLibre Challenge – Clasificación de Condición del Producto  
**Predicción de productos Nuevos vs Usados**

## 📌 Objetivo
El objetivo de este proyecto es construir un pipeline de *machine learning* robusto y reproducible que permita predecir si un producto publicado en MercadoLibre es **nuevo o usado**, a partir de información del aviso, del vendedor, del producto y *features* derivadas.

La solución pone foco en:
- Calidad de datos y prevención de *data leakage*
- Feature engineering sólido
- Comparación y optimización de modelos
- Modularidad y reproducibilidad

---

## 🧠 Visión General del Proyecto

El proyecto se organiza en un **pipeline en tres etapas**:

1. **Construcción del dataset** a partir de datos crudos (formato JSON) 
2. **Pipeline de Feature Engineering**  
3. **Entrenamiento, evaluación e inferencia del modelo final**

Cada etapa está implementada como un **script ejecutable independiente**, priorizando claridad, facilidad de debugging y control total del flujo.

---

## 📁 Estructura del proyecto

```text
MeLi_challenge/
│
├── .venv/                       # Virtual environment del proyecto
│
├── data/
│   ├── raw/                     # Dataset original (jsonlines)
│   ├── processed/               # Datasets base y datasets con Feature Engineering
│   └── artifacts/               # Modelos entrenados, logs, métricas y gráficos
│
├── src/
│   ├── data_process/
│   │   ├── __init__.py
│   │   ├── read_utils.py        # Aqui se incluye la función provista en el enunciado para parsear JSONs
│   │   └── build_dataset.py     # Orquesta la transformación de JSONs a DataFrame
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feat_eng_utils.py    # Define transformers y encoders
│   │   └── feat_eng_pipeline.py # Orquesta el proceso de Feature Engineering
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── model_experiments_cv.py  # Comparación de modelos base (RF, LGBM, XGB, CatBoost)
│   │   ├── xgb_optimize_hp.py       # Optimización de hiperparámetros con Optuna
│   │   └── xgb_select_thesshold.py  # Selección del punto de corte óptimo
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   └── train_infer_pipeline.py  # Entrenamiento final e inferencia sobre test
│
├── notebooks/
│   └── 01_eda.ipynb             # Análisis Exploratorio de Datos (EDA)
│
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Documentación principal
```

---

## 📊 Análisis Exploratorio de Datos (EDA)

El EDA se realizó con los siguientes objetivos:
- Comprender la estructura y calidad de los datos
- Detectar variables relevantes con poder predictivo
- Guiar decisiones de *feature engineering*

### Principales hallazgos:
- Variables con **más del 80% de valores nulos** fueron eliminadas
- Algunas variables (ej. `warranty`) mostraron buena relación con el target pese a tener alta tasa de nulos
- Gran parte de las variables son **categóricas**, muchas con **altísima cardinalidad**
- Se identificaron columnas que contienen **listas o diccionarios**, requiriendo procesamiento específico
- La mayoría de las variables numéricas presentan **distribuciones altamente sesgadas**, por lo que se aplicaron transformaciones logarítmicas
- Se detectaron columnas con **varianza nula o casi nula**, que fueron eliminadas por no aportar información predictiva

---

## 🧱 Feature Engineering

Se desarrolló un `FeatureEngineeringPipeline` propio con las siguientes características:

- Normalización de tipos (categóricas, booleanas, timestamps)
- Imputación de nulos:
  - Mediana para variables numéricas
  - Moda para categóricas y booleanas
- Generación de features a partir de:
  - Tags
  - Imágenes
  - Títulos
  - Variables temporales
  - Ratios y transformaciones numéricas
- Estrategias de encoding:
  - One-Hot Encoding (cardinalidad baja)
  - Frequency Encoding (cardinalidad alta sin relación muy diferente con el target entre sus categorías)
  - Target Encoding (cardinalidad alta mostrando relación muy diferente con el target entre sus categorías)
- Estricto control de *data leakage*:
  - El pipeline se **fitea solo con datos de train**
  - Los folds de validación se transforman sin refit

El pipeline es reutilizable en:
- Cross-validation
- Optimización de hiperparámetros
- Entrenamiento final productivo

---

## 🤖 Selección y Evaluación de Modelos

Se evaluaron los siguientes algoritmos usando **Stratified K-Fold Cross Validation**:

- Random Forest
- LightGBM
- CatBoost
- XGBoost

### Métricas evaluadas:
- **Accuracy** (restricción principal)
- **ROC-AUC** (métrica secundaria de optimización)
- **Recall de la clase “used”** (crítica para el negocio)
- **Precision de la clase “used”**

### Criterio de selección:
- Accuracy ≥ **0.86**
- Máximo ROC-AUC
- Buen balance entre recall y precision de usados

➡️ **XGBoost** resultó el mejor modelo en términos de performance global y estabilidad.

---

## 🔎 Optimización de Hiperparámetros

La optimización se realizó con **Optuna**, utilizando:

- 4 folds de CV por trial
- Objetivo: maximizar **ROC-AUC promedio**
- Restricción: descartar trials con accuracy < 0.86
- 20 trials
  
Los resultados fueron logueados y exportados para trazabilidad completa.

---

## 🎯 Optimización del Punto de Corte

Una vez entrenado el modelo final:
- Se optimizó el **threshold de decisión**
- Objetivo: maximizar el recall de productos usados
- Restricción: mantener accuracy ≥ 0.86

Esto permite priorizar la detección de productos usados sin degradar la calidad global del modelo.

---

## 🚀 Ejecución del Pipeline Final

### Paso 1 – Construcción del dataset
```bash
python src/data_process/build_dataset.py
```
### Paso 2 – Feature Engineering
```bash
python src/features/feat_eng_pipeline.py
```

### Paso 3 – Construcción del dataset
```bash
python src/model/train_infer_pipeline.py
```

## Resultados sobre el set de Test
- Accuracy: 0.865 --> superando el umbral requerido de 0.86
- ROC-AUC: 0.9465 --> indicando una excelente capacidad del modelo para discriminar las clases
- Recall: 0.9234 --> capturando la mayoria de los productos usados
- Precision: 0.8095 --> reflejando predicciones de calidad



