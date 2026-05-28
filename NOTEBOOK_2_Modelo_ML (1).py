# Databricks notebook source
# MAGIC %md
# MAGIC # 🤖 NOTEBOOK 2 — Construcción del Modelo de Machine Learning
# MAGIC ## Semáforo de Riesgo Vial · Caso 05
# MAGIC
# MAGIC **¿Qué aprenderás aquí?**
# MAGIC - Cargar datos limpios desde Delta Table
# MAGIC - Entender qué son features (X) y qué es el target (y)
# MAGIC - Dividir datos en entrenamiento y prueba (train/test split)
# MAGIC - Construir dos modelos supervisados: Random Forest y XGBoost
# MAGIC - Evaluar los modelos con métricas reales
# MAGIC - Seleccionar el mejor modelo
# MAGIC
# MAGIC **Modelos supervisados que usaremos:**
# MAGIC ```
# MAGIC RL (Regresión Logística) → baseline simple
# MAGIC Árboles (Random Forest)  → modelo principal
# MAGIC XGBoost                  → modelo avanzado (ganador)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ PASO 1 — Instalar XGBoost

# COMMAND ----------

# En Databricks Community necesitas instalar xgboost si no está disponible
# Ejecuta esta celda solo una vez
%pip install xgboost --quiet
print("✅ xgboost instalado")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📥 PASO 2 — Cargar el dataset limpio desde el Notebook 1

# COMMAND ----------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Opción A: Cargar desde Delta Table (si corriste el Notebook 1)
try:
    df = spark.table("siniestros_limpios").toPandas()
    print(f"✅ Datos cargados desde Delta Table: {df.shape}")
except:
    # Opción B: Cargar desde CSV (alternativa)
    df = pd.read_csv("/dbfs/FileStore/siniestros_limpios.csv")
    print(f"✅ Datos cargados desde CSV: {df.shape}")

print(f"\n📋 Columnas disponibles ({len(df.columns)}):")
print(list(df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 PASO 3 — Seleccionar Features (X) y Target (y)
# MAGIC
# MAGIC **¿Qué es una feature?**
# MAGIC Es una variable de entrada que el modelo usa para aprender.
# MAGIC Ejemplo: lluvia_mm, flujo_vehicular, edad_conductor.
# MAGIC
# MAGIC **¿Qué es el target?**
# MAGIC Es lo que queremos predecir: el nivel_riesgo (ALTO / MEDIO / BAJO).

# COMMAND ----------

# Features (variables de entrada al modelo)
FEATURES = [
    # Numéricas originales
    "lluvia_mm",          # ¿Cuánto llovió?
    "visibilidad_km",     # ¿Qué tan buena era la visibilidad?
    "flujo_vehicular",    # ¿Cuántos vehículos circulaban?
    "edad_conductor",     # Edad del conductor
    "experiencia_anios",  # Años manejando
    "infracciones_prev",  # Historial de infracciones
    "antiguedad_veh",     # Años del vehículo
    "temperatura_c",      # Temperatura al momento del siniestro
    "km_ruta",            # Kilómetro de la ruta

    # Features creadas en Notebook 1 (Feature Engineering)
    "es_horario_riesgo",      # 1=noche/madrugada, 0=día
    "indice_clima_peligro",   # Índice combinado de lluvia + visibilidad
    "ratio_exp_edad",         # Experiencia relativa al conductor
    "vehiculo_antiguo",       # 1=más de 10 años, 0=nuevo

    # Categóricas codificadas
    "tipo_via_cod",           # Nacional=?, Secundaria=?...
    "tipo_vehiculo_cod",      # Camión=?, Auto=?...
    "densidad_via_cod",       # Baja=?, Alta=?...
    "region_cod",             # Lima=?, Cusco=?...
    "estacion_cod",           # Verano=?, Invierno=?...
    "franja_horaria_cod",     # Mañana=?, Noche=?...
]

TARGET = "nivel_riesgo"  # BAJO / MEDIO / ALTO

# Filtrar solo las columnas que existen en el dataframe
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]
y = df[TARGET]

print(f"✅ Features seleccionadas : {len(FEATURES)}")
print(f"✅ Registros totales      : {len(X):,}")
print(f"\n📊 Distribución del target:")
print(y.value_counts())
print(f"\nPorcentajes:")
print(y.value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Codificar el target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Mapeo de clases: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# ✅ NUEVO: calcular pesos para compensar el desbalance
sample_weights_train = compute_sample_weight(class_weight="balanced", y=y_train)

print(f"\n✅ Train: {X_train.shape[0]:,} registros")
print(f"✅ Test : {X_test.shape[0]:,} registros")
print(f"\n📊 Distribución en Train:")
for cls, cnt in zip(le_target.classes_, np.bincount(y_train)):
    print(f"   {cls}: {cnt} ({cnt/len(y_train)*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✂️ PASO 4 — Dividir en Train y Test (80% / 20%)
# MAGIC
# MAGIC **¿Por qué dividir?**
# MAGIC - **Train (entrenamiento)**: El modelo aprende con este 80%
# MAGIC - **Test (prueba)**: Evaluamos con el 20% que el modelo NUNCA vio
# MAGIC
# MAGIC Esto simula lo que pasará en producción:
# MAGIC el modelo verá siniestros nuevos que no conoce.

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Codificar el target: BAJO=0, MEDIO=1, ALTO=2
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Mapeo de clases: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Dividir: 80% train, 20% test
# stratify=y_encoded garantiza que la proporción BAJO/MEDIO/ALTO sea igual en ambos sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print(f"\n✅ Train: {X_train.shape[0]:,} registros ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"✅ Test : {X_test.shape[0]:,} registros  ({X_test.shape[0]/len(X)*100:.0f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🌲 PASO 5 — Modelo 1: Regresión Logística (Baseline)
# MAGIC
# MAGIC Comenzamos con el modelo más simple para tener una **línea base**.
# MAGIC Si los modelos avanzados no superan esto, algo está mal.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# La Regresión Logística necesita que los datos estén estandarizados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # ¡Solo transform, NO fit!

# Crear y entrenar
modelo_rl = LogisticRegression(
    max_iter=1000,       # Iteraciones máximas para converger
    random_state=42,
    class_weight="balanced"  # Importante si hay desbalance de clases
)
modelo_rl.fit(X_train_scaled, y_train)

# Evaluar
y_pred_rl = modelo_rl.predict(X_test_scaled)
acc_rl = accuracy_score(y_test, y_pred_rl)

print(f"🔵 REGRESIÓN LOGÍSTICA — Accuracy: {acc_rl*100:.1f}%")
print("\nReporte detallado:")
print(classification_report(y_test, y_pred_rl, target_names=le_target.classes_))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🌳 PASO 6 — Modelo 2: Random Forest
# MAGIC
# MAGIC **¿Qué es Random Forest?**
# MAGIC Es un "bosque" de muchos árboles de decisión.
# MAGIC Cada árbol aprende un poco diferente (con datos aleatorios).
# MAGIC Al final, todos votan y gana la mayoría.
# MAGIC
# MAGIC Es mucho más poderoso que la Regresión Logística.

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

modelo_rf = RandomForestClassifier(
    n_estimators=200,    # 200 árboles en el bosque
    max_depth=12,        # Profundidad máxima de cada árbol
    min_samples_leaf=5,  # Mínimo de muestras en cada hoja
    random_state=42,
    n_jobs=-1,           # Usar todos los núcleos del CPU
    class_weight="balanced"
)

print("⏳ Entrenando Random Forest (200 árboles)...")
modelo_rf.fit(X_train, y_train)

y_pred_rf = modelo_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"\n🌲 RANDOM FOREST — Accuracy: {acc_rf*100:.1f}%")
print("\nReporte detallado:")
print(classification_report(y_test, y_pred_rf, target_names=le_target.classes_))

# COMMAND ----------

from xgboost import XGBClassifier

modelo_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,      # ✅ nuevo: evita overfitting
    gamma=0.1,               # ✅ nuevo: regularización
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss",
    verbosity=0
)

print("⏳ Entrenando XGBoost con pesos balanceados...")
modelo_xgb.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,   # ✅ clave: pesos por clase
    eval_set=[(X_test, y_test)],
    verbose=50
)

y_pred_xgb = modelo_xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"\n⚡ XGBOOST — Accuracy: {acc_xgb*100:.1f}%")
print(classification_report(y_test, y_pred_xgb, target_names=le_target.classes_))

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚡ PASO 7 — Modelo 3: XGBoost
# MAGIC
# MAGIC **¿Qué es XGBoost?**
# MAGIC Es un algoritmo de "Gradient Boosting": construye árboles de forma
# MAGIC secuencial donde cada árbol nuevo corrige los errores del anterior.
# MAGIC
# MAGIC Es el algoritmo más usado en competencias de ML y en producción.

# COMMAND ----------

from xgboost import XGBClassifier

modelo_xgb = XGBClassifier(
    n_estimators=300,    # 300 árboles secuenciales
    max_depth=6,         # Profundidad de cada árbol
    learning_rate=0.1,   # Qué tan rápido aprende (más bajo = más preciso pero lento)
    subsample=0.8,       # Usar 80% de los datos por árbol (reduce overfitting)
    colsample_bytree=0.8,# Usar 80% de features por árbol
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss",
    verbosity=0
)

print("⏳ Entrenando XGBoost (300 iteraciones)...")
modelo_xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # Monitorear en test durante entrenamiento
    verbose=50                     # Mostrar progreso cada 50 iteraciones
)

y_pred_xgb = modelo_xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"\n⚡ XGBOOST — Accuracy: {acc_xgb*100:.1f}%")
print("\nReporte detallado:")
print(classification_report(y_test, y_pred_xgb, target_names=le_target.classes_))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 PASO 8 — Comparar los 3 Modelos

# COMMAND ----------

from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

def evaluar_modelo(nombre, y_true, y_pred, modelo, X_eval):
    """Calcula múltiples métricas para un modelo."""
    acc   = accuracy_score(y_true, y_pred)
    f1_m  = f1_score(y_true, y_pred, average="macro")
    f1_w  = f1_score(y_true, y_pred, average="weighted")

    # AUC multi-clase (necesita probabilidades)
    try:
        y_prob = modelo.predict_proba(X_eval)
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except:
        auc = None

    return {"Modelo": nombre, "Accuracy": acc, "F1-Macro": f1_m,
            "F1-Weighted": f1_w, "AUC": auc}

resultados = pd.DataFrame([
    evaluar_modelo("Regresión Logística", y_test, y_pred_rl,  modelo_rl,  X_test_scaled),
    evaluar_modelo("Random Forest",       y_test, y_pred_rf,  modelo_rf,  X_test),
    evaluar_modelo("XGBoost",             y_test, y_pred_xgb, modelo_xgb, X_test),
])

resultados_pct = resultados.copy()
for col in ["Accuracy", "F1-Macro", "F1-Weighted", "AUC"]:
    resultados_pct[col] = resultados_pct[col].apply(
        lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
    )

print("=" * 60)
print("📊 COMPARATIVA DE MODELOS")
print("=" * 60)
display(resultados_pct)

mejor = resultados.loc[resultados["AUC"].idxmax(), "Modelo"]
print(f"\n🏆 MEJOR MODELO: {mejor}")

# COMMAND ----------

# DIAGNÓSTICO — Ejecuta esto antes de continuar
print("📊 Distribución del target:")
print(y.value_counts())
print()
print(y.value_counts(normalize=True).mul(100).round(1))

# COMMAND ----------

resultados = pd.DataFrame([
    evaluar_modelo("Regresión Logística", y_test, y_pred_rl,  modelo_rl,  X_test_scaled),
    evaluar_modelo("Random Forest",       y_test, y_pred_rf,  modelo_rf,  X_test),
    evaluar_modelo("XGBoost",             y_test, y_pred_xgb, modelo_xgb, X_test),
])

resultados_pct = resultados.copy()
for col in ["Accuracy", "F1-Macro", "F1-Weighted", "AUC"]:
    resultados_pct[col] = resultados_pct[col].apply(
        lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
    )

display(resultados_pct)

# ✅ CORREGIDO: usar F1-Weighted como criterio (más robusto con desbalance)
mejor = resultados.loc[resultados["F1-Weighted"].idxmax(), "Modelo"]
print(f"\n🏆 MEJOR MODELO (por F1-Weighted): {mejor}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔍 PASO 9 — Importancia de Variables (Feature Importance)
# MAGIC
# MAGIC ¿Qué variables importan más para predecir el riesgo?
# MAGIC Esto ayuda a entender qué datos son más valiosos.

# COMMAND ----------

# XGBoost tiene su propio método de importancia
importancias = pd.DataFrame({
    "feature"    : FEATURES,
    "importancia": modelo_xgb.feature_importances_
}).sort_values("importancia", ascending=False)

print("📊 Top 10 variables más importantes (XGBoost):")
print(importancias.head(10).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 PASO 10 — Guardar el Modelo
# MAGIC
# MAGIC Guardamos el modelo entrenado para usarlo en el Notebook 3 (scoring).

# COMMAND ----------

import pickle
import os

# Crear directorio
os.makedirs("/dbfs/FileStore/modelos", exist_ok=True)

# Guardar modelos
with open("/dbfs/FileStore/modelos/modelo_rf.pkl",     "wb") as f: pickle.dump(modelo_rf,  f)
with open("/dbfs/FileStore/modelos/modelo_xgb.pkl",    "wb") as f: pickle.dump(modelo_xgb, f)
with open("/dbfs/FileStore/modelos/le_target.pkl",     "wb") as f: pickle.dump(le_target,  f)
with open("/dbfs/FileStore/modelos/features_list.pkl", "wb") as f: pickle.dump(FEATURES,   f)

# Guardar también las predicciones del test set
df_resultados = X_test.copy()
df_resultados["y_real"]            = le_target.inverse_transform(y_test)
df_resultados["y_pred_rf"]         = le_target.inverse_transform(y_pred_rf)
df_resultados["y_pred_xgb"]        = le_target.inverse_transform(y_pred_xgb)
df_resultados["proba_xgb_ALTO"]    = modelo_xgb.predict_proba(X_test)[:, 2]
df_resultados["proba_xgb_MEDIO"]   = modelo_xgb.predict_proba(X_test)[:, 1]
df_resultados["proba_xgb_BAJO"]    = modelo_xgb.predict_proba(X_test)[:, 0]

spark.createDataFrame(df_resultados).write.format("delta").mode("overwrite").saveAsTable("resultados_modelo")

print("✅ Modelos guardados en /dbfs/FileStore/modelos/")
print("✅ Tabla resultados_modelo guardada en Delta")

# COMMAND ----------

# Paso 1: Ver qué catálogos tienes disponibles
print("📋 Catálogos disponibles en tu cuenta:")
display(spark.sql("SHOW CATALOGS"))

# COMMAND ----------

import pickle, os

# Usar catálogo 'workspace' (el tuyo en Databricks Community)
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.semaforo")
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.semaforo.archivos")
BASE_PATH = "/Volumes/workspace/semaforo/archivos"

print(f"✅ Volume listo: {BASE_PATH}")

# Guardar modelos
objetos = {
    "modelo_rf.pkl"     : modelo_rf,
    "modelo_xgb.pkl"    : modelo_xgb,
    "le_target.pkl"     : le_target,
    "features_list.pkl" : FEATURES,
}

for nombre, objeto in objetos.items():
    ruta = f"{BASE_PATH}/{nombre}"
    with open(ruta, "wb") as f:
        pickle.dump(objeto, f)
    print(f"✅ Guardado: {ruta}")

# Guardar ruta base para que Notebook 3 la encuentre
with open(f"{BASE_PATH}/base_path.txt", "w") as f:
    f.write(BASE_PATH)

print(f"\n🎯 Modelos guardados correctamente.")
print(f"   En Notebook 3 usa BASE_PATH = '{BASE_PATH}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ ¡NOTEBOOK 2 COMPLETADO!
# MAGIC
# MAGIC **Lo que hiciste:**
# MAGIC - ✅ Cargaste los datos limpios del Notebook 1
# MAGIC - ✅ Definiste features (X) y target (y = BAJO/MEDIO/ALTO)
# MAGIC - ✅ Dividiste en Train 80% / Test 20%
# MAGIC - ✅ Entrenaste 3 modelos: RL, Random Forest, XGBoost
# MAGIC - ✅ Comparaste métricas y elegiste el mejor (XGBoost)
# MAGIC - ✅ Analizaste la importancia de variables
# MAGIC - ✅ Guardaste los modelos para producción
# MAGIC
# MAGIC **➡️ Siguiente paso: Abre el NOTEBOOK_3_Entrenamiento_Scoring.py**