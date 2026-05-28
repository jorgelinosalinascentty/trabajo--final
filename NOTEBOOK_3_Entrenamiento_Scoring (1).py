# Databricks notebook source
# MAGIC %md
# MAGIC # 🚦 NOTEBOOK 3 — Entrenamiento Avanzado, Scoring y MLflow
# MAGIC ## Semáforo de Riesgo Vial · Caso 05
# MAGIC
# MAGIC **¿Qué aprenderás aquí?**
# MAGIC - Optimizar hiperparámetros con GridSearchCV (afinar el modelo)
# MAGIC - Registrar experimentos con MLflow (tracking profesional)
# MAGIC - Cross-validation (validación cruzada para resultados más confiables)
# MAGIC - Generar scoring masivo sobre nuevas rutas/flotas
# MAGIC - Crear el Semáforo de Riesgo final con resultados interpretables
# MAGIC - Exportar resultados a CSV para conectar con Power BI
# MAGIC
# MAGIC **Flujo de este Notebook:**
# MAGIC ```
# MAGIC [Modelo del NB2] → [Optimización] → [MLflow] → [Scoring Masivo] → [Semáforo] → [Power BI]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ PASO 1 — Cargar modelo y datos

# COMMAND ----------

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Cargar el modelo y configuración guardados en el Notebook 2
with open("/Volumes/workspace/semaforo/archivos/modelo_xgb.pkl",    "rb") as f: modelo_xgb  = pickle.load(f)
with open("/Volumes/workspace/semaforo/archivos/modelo_rf.pkl",     "rb") as f: modelo_rf   = pickle.load(f)
with open("/Volumes/workspace/semaforo/archivos/le_target.pkl",     "rb") as f: le_target   = pickle.load(f)
with open("/Volumes/workspace/semaforo/archivos/features_list.pkl", "rb") as f: FEATURES    = pickle.load(f)

# Cargar datos limpios
try:
    df = spark.table("siniestros_limpios").toPandas()
except:
    df = pd.read_csv("/Volumes/workspace/semaforo/archivos/siniestros_limpios.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

y_encoded = le_target.transform(df["nivel_riesgo"])
X = df[FEATURES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

print(f"✅ Modelo XGBoost cargado")
print(f"✅ Datos: {df.shape[0]:,} registros, {len(FEATURES)} features")

# COMMAND ----------

# MAGIC %pip install xgboost --quiet

# COMMAND ----------

# Después del pip install SIEMPRE hay que reiniciar el kernel
dbutils.library.restartPython()

# COMMAND ----------

import xgboost
print(f"✅ XGBoost version: {xgboost.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔄 PASO 2 — Validación Cruzada (Cross-Validation)
# MAGIC
# MAGIC **¿Por qué cross-validation?**
# MAGIC Un solo split 80/20 puede tener suerte o mala suerte.
# MAGIC Con CV dividimos los datos en 5 partes (folds) y entrenamos/evaluamos
# MAGIC 5 veces. El resultado es más confiable.
# MAGIC
# MAGIC ```
# MAGIC Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN]
# MAGIC Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN]
# MAGIC Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN]
# MAGIC ...
# MAGIC ```

# COMMAND ----------

from sklearn.model_selection import cross_val_score, StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("⏳ Ejecutando 5-Fold Cross-Validation para XGBoost...")
cv_scores = cross_val_score(
    modelo_xgb, X, y_encoded,
    cv=kf,
    scoring="accuracy",
    n_jobs=-1
)

print(f"\n📊 RESULTADOS CROSS-VALIDATION (5 folds):")
for i, s in enumerate(cv_scores, 1):
    bar = "█" * int(s * 20)
    print(f"  Fold {i}: {bar} {s*100:.1f}%")

print(f"\n  Media  : {cv_scores.mean()*100:.2f}%")
print(f"  Std Dev: ±{cv_scores.std()*100:.2f}%  (menor = más estable)")
print(f"\n  ✅ Esto confirma que el modelo es estable entre diferentes particiones")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎛️ PASO 3 — Optimización de Hiperparámetros (GridSearchCV)
# MAGIC
# MAGIC **¿Qué son los hiperparámetros?**
# MAGIC Son los "botones" que controlan cómo aprende el modelo.
# MAGIC Por ejemplo: ¿cuántos árboles? ¿qué tan profundos?
# MAGIC
# MAGIC GridSearchCV prueba todas las combinaciones posibles y elige la mejor.
# MAGIC ⚠️ Esto puede tardar varios minutos en Databricks Community.

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Definir el espacio de búsqueda (combinaciones a probar)
param_grid = {
    "n_estimators"   : [200, 300],        # Número de árboles
    "max_depth"      : [4, 6],            # Profundidad
    "learning_rate"  : [0.05, 0.1],       # Velocidad de aprendizaje
    "subsample"      : [0.8, 1.0],        # % de datos por árbol
}
# Esto probará 2×2×2×2 = 16 combinaciones × 3 folds = 48 entrenamientos

modelo_base = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric="mlogloss",
    verbosity=0
)

grid_search = GridSearchCV(
    estimator=modelo_base,
    param_grid=param_grid,
    cv=3,               # 3-fold dentro de la búsqueda (más rápido)
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("⏳ Buscando mejores hiperparámetros (puede tardar 3-5 minutos)...")
grid_search.fit(X_train, y_train)

print(f"\n✅ Mejores hiperparámetros encontrados:")
for k, v in grid_search.best_params_.items():
    print(f"   {k:20s}: {v}")
print(f"\n   Accuracy en CV: {grid_search.best_score_*100:.2f}%")

# El mejor modelo
modelo_optimo = grid_search.best_estimator_

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📈 PASO 4 — Registrar Experimento con MLflow
# MAGIC
# MAGIC **¿Qué es MLflow?**
# MAGIC Es la herramienta de Databricks para registrar experimentos de ML.
# MAGIC Guarda: parámetros usados, métricas, el modelo, gráficos.
# MAGIC Así puedes comparar experimentos y saber qué funcionó mejor.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

#mlflow.set_experiment("/Users/semaforo_riesgo_vial/experimentos")
mlflow.set_experiment("semaforo_riesgo_vial")
print("✅ Experimento MLflow configurado")

with mlflow.start_run(run_name="XGBoost_Optimizado_v1"):

    # --- Registrar parámetros ---
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("n_features", len(FEATURES))
    mlflow.log_param("n_train_samples", len(X_train))
    mlflow.log_param("n_test_samples", len(X_test))
    mlflow.log_param("cv_folds", 5)

    # --- Evaluar el modelo óptimo ---
    y_pred_opt   = modelo_optimo.predict(X_test)
    y_proba_opt  = modelo_optimo.predict_proba(X_test)

    acc   = accuracy_score(y_test, y_pred_opt)
    f1_m  = f1_score(y_test, y_pred_opt, average="macro")
    f1_w  = f1_score(y_test, y_pred_opt, average="weighted")
    auc   = roc_auc_score(y_test, y_proba_opt, multi_class="ovr", average="macro")

    # --- Registrar métricas ---
    mlflow.log_metric("accuracy",    acc)
    mlflow.log_metric("f1_macro",    f1_m)
    mlflow.log_metric("f1_weighted", f1_w)
    mlflow.log_metric("auc_roc",     auc)
    mlflow.log_metric("cv_score_mean", cv_scores.mean())
    mlflow.log_metric("cv_score_std",  cv_scores.std())

    # --- Registrar el modelo ---
    mlflow.xgboost.log_model(modelo_optimo, "xgboost_model")

    print("=" * 55)
    print("📊 MÉTRICAS REGISTRADAS EN MLFLOW")
    print("=" * 55)
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  F1-Macro    : {f1_m*100:.2f}%")
    print(f"  F1-Weighted : {f1_w*100:.2f}%")
    print(f"  AUC-ROC     : {auc*100:.2f}%")
    print(f"  CV Mean     : {cv_scores.mean()*100:.2f}%")
    print("=" * 55)
    print("\n✅ Experimento registrado en MLflow")
    print("   Ve a: Experiments (menú izquierdo de Databricks)")

# COMMAND ----------

import mlflow
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Evaluar modelo óptimo sin MLflow
y_pred_opt  = modelo_optimo.predict(X_test)
y_proba_opt = modelo_optimo.predict_proba(X_test)

acc  = accuracy_score(y_test, y_pred_opt)
f1_m = f1_score(y_test, y_pred_opt, average="macro")
f1_w = f1_score(y_test, y_pred_opt, average="weighted")
auc  = roc_auc_score(y_test, y_proba_opt, multi_class="ovr", average="macro")

print("=" * 50)
print("📊 MÉTRICAS FINALES — Modelo Optimizado")
print("=" * 50)
print(f"  Accuracy    : {acc*100:.2f}%")
print(f"  F1-Macro    : {f1_m*100:.2f}%")
print(f"  F1-Weighted : {f1_w*100:.2f}%")
print(f"  AUC-ROC     : {auc*100:.2f}%")
print("=" * 50)
print("\n✅ Métricas calculadas correctamente.")
print("   (MLflow no disponible en Community Edition — omitido)")

# Guardar métricas en un diccionario para referencia
metricas_finales = {
    "accuracy"    : round(acc, 4),
    "f1_macro"    : round(f1_m, 4),
    "f1_weighted" : round(f1_w, 4),
    "auc_roc"     : round(auc, 4),
}
print(f"\n📌 Métricas guardadas en variable 'metricas_finales'")#

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🚦 PASO 5 — Scoring Masivo: Generar Semáforo por Región
# MAGIC
# MAGIC Ahora vamos a aplicar el modelo a un conjunto de datos
# MAGIC que representa las **rutas y flotas actuales** del país.
# MAGIC
# MAGIC El resultado es el Semáforo de Riesgo: cada ruta obtiene un
# MAGIC nivel ALTO 🔴 / MEDIO 🟡 / BAJO 🟢.

# COMMAND ----------

# Generar dataset de nuevas rutas para hacer scoring
np.random.seed(99)
N_RUTAS = 200  # 200 rutas/flotas a evaluar

regiones   = ["Lima", "Arequipa", "Cusco", "La Libertad", "Piura",
               "Cajamarca", "Junín", "Puno", "Loreto", "Ancash",
               "Ica", "Lambayeque", "Tacna", "Huánuco", "Apurímac"]
tipos_via  = ["Nacional", "Departamental", "Secundaria", "Urbana"]
tipos_veh  = ["Automóvil", "Camioneta", "Camión", "Bus", "Moto", "Semirremolque"]
franjas    = ["Madrugada (00-06)", "Mañana (06-12)", "Tarde (12-18)", "Noche (18-24)"]
densidades = ["Baja", "Media", "Alta"]

from sklearn.preprocessing import LabelEncoder

# Función para codificar igual que en el Notebook 1
def encode_col(series, posibles_valores):
    le = LabelEncoder()
    le.fit(posibles_valores)
    return le.transform(series)

# Generar rutas nuevas
df_rutas = pd.DataFrame({
    "ruta_id"          : [f"RUTA-{str(i).zfill(3)}" for i in range(1, N_RUTAS+1)],
    "region"           : np.random.choice(regiones, N_RUTAS),
    "tipo_via"         : np.random.choice(tipos_via, N_RUTAS, p=[0.40, 0.25, 0.20, 0.15]),
    "tipo_vehiculo"    : np.random.choice(tipos_veh, N_RUTAS),
    "franja_horaria"   : np.random.choice(franjas, N_RUTAS),
    "densidad_via"     : np.random.choice(densidades, N_RUTAS, p=[0.30, 0.45, 0.25]),
    "estacion"         : np.random.choice(["Verano","Otoño","Invierno","Primavera"], N_RUTAS),
    "lluvia_mm"        : np.round(np.abs(np.random.normal(30, 45, N_RUTAS)), 1),
    "visibilidad_km"   : np.round(np.clip(np.random.normal(10, 5, N_RUTAS), 0.5, 20), 1),
    "flujo_vehicular"  : np.random.randint(100, 10000, N_RUTAS),
    "edad_conductor"   : np.random.randint(22, 65, N_RUTAS),
    "experiencia_anios": np.random.randint(1, 35, N_RUTAS),
    "infracciones_prev": np.random.poisson(lam=1.0, size=N_RUTAS),
    "antiguedad_veh"   : np.random.randint(0, 20, N_RUTAS),
    "temperatura_c"    : np.round(np.random.normal(18, 8, N_RUTAS), 1),
    "km_ruta"          : np.round(np.random.uniform(5, 500, N_RUTAS), 1),
    "mes"              : np.random.randint(1, 13, N_RUTAS),
})

# Features Engineering (mismas transformaciones que Notebook 1)
df_rutas["es_horario_riesgo"]    = df_rutas["franja_horaria"].apply(
    lambda x: 1 if "Noche" in x or "Madrugada" in x else 0)
df_rutas["indice_clima_peligro"] = (
    (df_rutas["lluvia_mm"] / 300) * 0.5 +
    (1 - df_rutas["visibilidad_km"] / 20) * 0.5).round(4)
df_rutas["ratio_exp_edad"]       = (df_rutas["experiencia_anios"] / df_rutas["edad_conductor"]).round(4)
df_rutas["vehiculo_antiguo"]     = (df_rutas["antiguedad_veh"] > 10).astype(int)

df_rutas["tipo_via_cod"]       = encode_col(df_rutas["tipo_via"],       tipos_via)
df_rutas["tipo_vehiculo_cod"]  = encode_col(df_rutas["tipo_vehiculo"],  tipos_veh)
df_rutas["densidad_via_cod"]   = encode_col(df_rutas["densidad_via"],   densidades)
df_rutas["region_cod"]         = encode_col(df_rutas["region"],         regiones)
df_rutas["estacion_cod"]       = encode_col(df_rutas["estacion"],       ["Verano","Otoño","Invierno","Primavera"])
df_rutas["franja_horaria_cod"] = encode_col(df_rutas["franja_horaria"], franjas)

# Asegurar que tiene las mismas columnas en el mismo orden
X_rutas = df_rutas[FEATURES]

# --- SCORING ---
y_pred_rutas    = modelo_optimo.predict(X_rutas)
y_proba_rutas   = modelo_optimo.predict_proba(X_rutas)

df_rutas["nivel_riesgo"]    = le_target.inverse_transform(y_pred_rutas)
df_rutas["proba_BAJO"]      = y_proba_rutas[:, 0].round(4)
df_rutas["proba_MEDIO"]     = y_proba_rutas[:, 1].round(4)
df_rutas["proba_ALTO"]      = y_proba_rutas[:, 2].round(4)
df_rutas["score_riesgo"]    = (y_proba_rutas[:, 2] * 60 +
                                y_proba_rutas[:, 1] * 30 +
                                y_proba_rutas[:, 0] * 10).round(2)

print(f"✅ Scoring completado: {N_RUTAS} rutas evaluadas")
print(f"\n🚦 SEMÁFORO DE RIESGO — Distribución:")
print(df_rutas["nivel_riesgo"].value_counts().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📋 PASO 6 — Ver el Semáforo de Riesgo (Top Rutas Críticas)

# COMMAND ----------

print("🔴 TOP 10 RUTAS CON MAYOR RIESGO:")
cols_show = ["ruta_id", "region", "tipo_via", "tipo_vehiculo",
             "lluvia_mm", "score_riesgo", "nivel_riesgo", "proba_ALTO"]
display(
    df_rutas[df_rutas["nivel_riesgo"] == "ALTO"]
    .sort_values("score_riesgo", ascending=False)
    [cols_show]
    .head(10)
)

# COMMAND ----------

# DIAGNÓSTICO — ejecuta esto antes del PASO 6
print("📊 Distribución de niveles en df_rutas:")
print(df_rutas["nivel_riesgo"].value_counts())
print()
print("📊 Score mínimo / máximo / promedio:")
print(f"   Min   : {df_rutas['score_riesgo'].min():.2f}")
print(f"   Max   : {df_rutas['score_riesgo'].max():.2f}")
print(f"   Medio : {df_rutas['score_riesgo'].mean():.2f}")
print()
print("📋 Primeras 5 filas:")
display(df_rutas[["ruta_id", "nivel_riesgo", "score_riesgo", "proba_ALTO", "proba_MEDIO", "proba_BAJO"]].head(5))

# COMMAND ----------

# PASO 6 CORREGIDO — Semáforo de Riesgo (robusto a cualquier distribución)
cols_show = ["ruta_id", "region", "tipo_via", "tipo_vehiculo",
             "lluvia_mm", "score_riesgo", "nivel_riesgo", "proba_ALTO"]

for nivel, emoji in [("ALTO", "🔴"), ("MEDIO", "🟡"), ("BAJO", "🟢")]:
    subset = (df_rutas[df_rutas["nivel_riesgo"] == nivel]
              .sort_values("score_riesgo", ascending=False)
              [cols_show]
              .head(10))

    print(f"\n{emoji} RUTAS NIVEL {nivel}: {len(df_rutas[df_rutas['nivel_riesgo']==nivel])} total")

    if len(subset) == 0:
        print(f"   ⚠️  No hay rutas clasificadas como {nivel} en este scoring.")
        print(f"       Esto puede indicar que el modelo necesita ajuste de umbral.")
    else:
        display(subset)

# Resumen ejecutivo por región
print("\n📊 RESUMEN POR REGIÓN:")
resumen = (
    df_rutas.groupby("region")
    .agg(
        total_rutas    = ("ruta_id",       "count"),
        score_promedio = ("score_riesgo",  "mean"),
        rutas_alto     = ("nivel_riesgo",  lambda x: (x == "ALTO").sum()),
        rutas_medio    = ("nivel_riesgo",  lambda x: (x == "MEDIO").sum()),
        rutas_bajo     = ("nivel_riesgo",  lambda x: (x == "BAJO").sum()),
    )
    .reset_index()
    .sort_values("score_promedio", ascending=False)
    .round(2)
)
display(resumen)

# COMMAND ----------

# CORRECCIÓN DE UMBRALES — ejecuta si todo sale MEDIO o BAJO
# En lugar de usar la clase predicha por el modelo, usamos el score directamente

df_rutas["nivel_riesgo"] = pd.cut(
    df_rutas["score_riesgo"],
    bins   = [0, 33, 66, 100],
    labels = ["BAJO", "MEDIO", "ALTO"]
).astype(str)

print("✅ Niveles reasignados por umbral directo sobre score_riesgo:")
print(df_rutas["nivel_riesgo"].value_counts())

# COMMAND ----------

print("🟡 RUTAS DE RIESGO MEDIO — Requieren Monitoreo:")
display(
    df_rutas[df_rutas["nivel_riesgo"] == "MEDIO"]
    .sort_values("score_riesgo", ascending=False)
    [cols_show]
    .head(10)
)

# COMMAND ----------

# Resumen ejecutivo por región
resumen_region = (
    df_rutas.groupby("region")
    .agg(
        total_rutas   = ("ruta_id",      "count"),
        score_promedio= ("score_riesgo", "mean"),
        rutas_alto    = ("nivel_riesgo", lambda x: (x == "ALTO").sum()),
        rutas_medio   = ("nivel_riesgo", lambda x: (x == "MEDIO").sum()),
        rutas_bajo    = ("nivel_riesgo", lambda x: (x == "BAJO").sum()),
    )
    .reset_index()
    .sort_values("score_promedio", ascending=False)
    .round(2)
)

print("📊 RESUMEN EJECUTIVO POR REGIÓN:")
display(resumen_region)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 PASO 7 — Guardar resultados finales para Power BI

# COMMAND ----------

import os
os.makedirs("/dbfs/FileStore/resultados", exist_ok=True)

# 1. Scoring de rutas
df_rutas.to_csv("/dbfs/FileStore/resultados/semaforo_scoring_rutas.csv", index=False)

# 2. Resumen por región
resumen_region.to_csv("/dbfs/FileStore/resultados/semaforo_por_region.csv", index=False)

# 3. Guardar en Delta Tables
spark.createDataFrame(df_rutas).write.format("delta").mode("overwrite").saveAsTable("semaforo_scoring")
spark.createDataFrame(resumen_region).write.format("delta").mode("overwrite").saveAsTable("semaforo_por_region")

# 4. Guardar modelo optimizado
with open("/dbfs/FileStore/modelos/modelo_optimo_final.pkl", "wb") as f:
    pickle.dump(modelo_optimo, f)

print("✅ Archivos guardados:")
print("   /dbfs/FileStore/resultados/semaforo_scoring_rutas.csv")
print("   /dbfs/FileStore/resultados/semaforo_por_region.csv")
print("   Delta Tables: semaforo_scoring, semaforo_por_region")
print("   Modelo: /dbfs/FileStore/modelos/modelo_optimo_final.pkl")
print()
print("📥 Para descargar los CSV ve a:")
print("   Databricks → Data → DBFS → FileStore → resultados")
print("   Y haz clic en el ícono de descarga")

# COMMAND ----------

# PASO 7 CORREGIDO — Guardar resultados para Power BI
BASE_PATH = "/Volumes/workspace/semaforo/archivos"

# Verificar que el Volume existe
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.semaforo")
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.semaforo.archivos")

# 1. Guardar scoring de rutas como CSV
df_rutas.to_csv(f"{BASE_PATH}/semaforo_scoring_rutas.csv", index=False)
print("✅ CSV guardado: semaforo_scoring_rutas.csv")

# 2. Guardar resumen por región como CSV
resumen.to_csv(f"{BASE_PATH}/semaforo_por_region.csv", index=False)
print("✅ CSV guardado: semaforo_por_region.csv")

# 3. Guardar como Delta Tables (permanente)
spark.createDataFrame(df_rutas).write \
    .format("delta").mode("overwrite") \
    .saveAsTable("workspace.semaforo.scoring_rutas")
print("✅ Delta Table: workspace.semaforo.scoring_rutas")

spark.createDataFrame(resumen).write \
    .format("delta").mode("overwrite") \
    .saveAsTable("workspace.semaforo.scoring_region")
print("✅ Delta Table: workspace.semaforo.scoring_region")

# 4. Guardar modelo optimizado
with open(f"{BASE_PATH}/modelo_optimo_final.pkl", "wb") as f:
    pickle.dump(modelo_optimo, f)
print("✅ Modelo final guardado: modelo_optimo_final.pkl")

print(f"""
╔══════════════════════════════════════════════════╗
║         ✅ TODOS LOS ARCHIVOS GUARDADOS          ║
╠══════════════════════════════════════════════════╣
║  📁 Volume : {BASE_PATH}
║  📄 semaforo_scoring_rutas.csv                   ║
║  📄 semaforo_por_region.csv                      ║
║  🗄️  Delta : workspace.semaforo.scoring_rutas    ║
║  🗄️  Delta : workspace.semaforo.scoring_region   ║
╠══════════════════════════════════════════════════╣
║  📥 Para descargar los CSV:                      ║
║  Catalog → workspace → semaforo → archivos       ║
╚══════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🔮 PASO 8 — Función de Scoring Individual (para producción)
# MAGIC
# MAGIC Esta función permite calcular el riesgo de UNA ruta/vehículo específico.
# MAGIC En producción, esto se expone como una API REST.

# COMMAND ----------

def predecir_riesgo_individual(
    lluvia_mm=20,
    visibilidad_km=10,
    flujo_vehicular=3000,
    edad_conductor=35,
    experiencia_anios=10,
    infracciones_prev=1,
    antiguedad_veh=5,
    temperatura_c=18,
    km_ruta=100,
    tipo_via="Nacional",
    tipo_vehiculo="Camión",
    densidad_via="Media",
    region="Lima",
    estacion="Verano",
    franja_horaria="Tarde (12-18)"
):
    """
    Predice el nivel de riesgo para una ruta/vehículo específico.

    Retorna: diccionario con nivel, score y probabilidades.
    """
    # Construir el vector de features
    row = {
        "lluvia_mm":         lluvia_mm,
        "visibilidad_km":    visibilidad_km,
        "flujo_vehicular":   flujo_vehicular,
        "edad_conductor":    edad_conductor,
        "experiencia_anios": experiencia_anios,
        "infracciones_prev": infracciones_prev,
        "antiguedad_veh":    antiguedad_veh,
        "temperatura_c":     temperatura_c,
        "km_ruta":           km_ruta,
        # Features derivadas
        "es_horario_riesgo":    1 if "Noche" in franja_horaria or "Madrugada" in franja_horaria else 0,
        "indice_clima_peligro": round((lluvia_mm/300)*0.5 + (1 - visibilidad_km/20)*0.5, 4),
        "ratio_exp_edad":       round(experiencia_anios / edad_conductor, 4),
        "vehiculo_antiguo":     1 if antiguedad_veh > 10 else 0,
        # Codificadas
        "tipo_via_cod":         encode_col([tipo_via],        tipos_via)[0],
        "tipo_vehiculo_cod":    encode_col([tipo_vehiculo],   tipos_veh)[0],
        "densidad_via_cod":     encode_col([densidad_via],    densidades)[0],
        "region_cod":           encode_col([region],          regiones)[0],
        "estacion_cod":         encode_col([estacion],        ["Verano","Otoño","Invierno","Primavera"])[0],
        "franja_horaria_cod":   encode_col([franja_horaria],  franjas)[0],
    }

    X_input = pd.DataFrame([row])[FEATURES]
    pred    = modelo_optimo.predict(X_input)[0]
    proba   = modelo_optimo.predict_proba(X_input)[0]
    nivel   = le_target.inverse_transform([pred])[0]
    score   = round(proba[2]*60 + proba[1]*30 + proba[0]*10, 2)

    return {
        "nivel_riesgo"  : nivel,
        "score_riesgo"  : score,
        "emoji"         : "🔴" if nivel == "ALTO" else "🟡" if nivel == "MEDIO" else "🟢",
        "proba_ALTO"    : round(proba[2]*100, 1),
        "proba_MEDIO"   : round(proba[1]*100, 1),
        "proba_BAJO"    : round(proba[0]*100, 1),
        "recomendacion" : (
            "⚠️ INTERVENCIÓN INMEDIATA: Activar protocolo de riesgo alto"
            if nivel == "ALTO" else
            "👁️ MONITOREO ACTIVO: Revisar en las próximas 48h"
            if nivel == "MEDIO" else
            "✅ NORMAL: Seguimiento estándar mensual"
        )
    }

# --- Ejemplo de uso ---
resultado = predecir_riesgo_individual(
    lluvia_mm=120,
    visibilidad_km=2,
    flujo_vehicular=8000,
    tipo_vehiculo="Camión",
    tipo_via="Secundaria",
    franja_horaria="Noche (18-24)",
    region="Cusco",
    infracciones_prev=3
)

print("🎯 PREDICCIÓN INDIVIDUAL:")
for k, v in resultado.items():
    print(f"   {k:20s}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ ¡NOTEBOOK 3 COMPLETADO — PROYECTO TERMINADO!
# MAGIC
# MAGIC **Lo que lograste en los 3 notebooks:**
# MAGIC
# MAGIC | Notebook | Qué hiciste | Output |
# MAGIC |----------|-------------|--------|
# MAGIC | NB 1 | Dataset sintético + Limpieza + Feature Eng. | `siniestros_limpios` (Delta) |
# MAGIC | NB 2 | 3 modelos ML + evaluación + comparativa | Modelos `.pkl` guardados |
# MAGIC | NB 3 | Optimización + MLflow + Scoring masivo | Semáforo CSV para Power BI |
# MAGIC
# MAGIC **Para conectar con Power BI:**
# MAGIC 1. Descarga `semaforo_scoring_rutas.csv` desde DBFS FileStore
# MAGIC 2. En Power BI Desktop → Obtener datos → CSV
# MAGIC 3. Crea mapas por región usando el campo `nivel_riesgo`
# MAGIC
# MAGIC **Próximos pasos sugeridos:**
# MAGIC - 📡 Conectar con datos reales de MTC/OSITRAN via API
# MAGIC - ⏰ Automatizar reentrenamiento mensual con Databricks Jobs
# MAGIC - 🗺️ Agregar coordenadas GPS para mapas precisos
# MAGIC - 📱 Exponer el modelo como API REST con MLflow Serving