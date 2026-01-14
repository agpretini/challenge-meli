# MercadoLibre Challenge – Clasificación de Condición del Producto  
**Predicción de productos Nuevos vs Usados**

## 📌 Objetivo
El objetivo de este proyecto es construir un pipeline de *machine learning* robusto y reproducible que permita predecir si un producto publicado en MercadoLibre es **nuevo o usado**, a partir de información estructurada del aviso y *features* derivadas.

La solución pone foco en:
- Calidad de datos y prevención de *data leakage*
- Feature engineering sólido
- Comparación y optimización de modelos
- Modularidad y reproducibilidad

---

## 🧠 Visión General del Proyecto

El proyecto se organiza en un **pipeline offline en tres etapas**:

1. **Construcción del dataset** a partir de datos crudos  
2. **Pipeline de Feature Engineering**  
3. **Entrenamiento, evaluación e inferencia del modelo final**

Cada etapa está implementada como un **script ejecutable independiente**, priorizando claridad, facilidad de debugging y control total del flujo.

---

## 📂 Estructura del Repositorio



---

## 📊 Análisis Exploratorio de Datos (EDA)

El EDA se realizó con los siguientes objetivos:
- Comprender la estructura y calidad de los datos
- Detectar variables relevantes con poder predictivo
- Guiar decisiones de *feature engineering*

### Principales hallazgos:
- Variables con **más del 90% de valores nulos** fueron eliminadas
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
  - Variables temporales
  - Ratios y transformaciones numéricas
- Estrategias de encoding:
  - One-Hot Encoding
  - Frequency Encoding
  - Target Encoding
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
- Ajuste explícito de `scale_pos_weight` para tratar el desbalance de clases

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

### Paso 2 – Feature Engineering
```bash
python src/features/feat_eng_pipeline.py

### Paso 3 – Construcción del dataset
```bash
python src/model/train_infer_pipeline.py




