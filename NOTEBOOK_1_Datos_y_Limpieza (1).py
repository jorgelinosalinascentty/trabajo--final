# Databricks notebook source
# MAGIC %md
# MAGIC # 🧹 NOTEBOOK 1 — Generación de Dataset y Limpieza de Datos
# MAGIC ## Semáforo de Riesgo Vial · Caso 05
# MAGIC
# MAGIC **¿Qué aprenderás aquí?**
# MAGIC - Cómo crear un dataset ficticio (sintético) representativo de siniestros viales
# MAGIC - Cómo hacer Feature Engineering: limpiar nulos, eliminar outliers, corregir formatos
# MAGIC - Cómo guardar los datos limpios en el Data Lake de Databricks (Delta Tables)
# MAGIC
# MAGIC **Flujo:**
# MAGIC ```
# MAGIC [Datos Sintéticos] → [Introducir errores realistas] → [Limpiar y transformar] → [Guardar en Delta]
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## ⚙️ PASO 1 — Instalar librerías necesarias
# MAGIC
# MAGIC Databricks Community ya tiene instalado pandas, numpy y scikit-learn.
# MAGIC Solo necesitamos verificar que todo esté disponible.

# COMMAND ----------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Verificar versiones
print(f"✅ pandas  : {pd.__version__}")
print(f"✅ numpy   : {np.__version__}")
print("✅ Librerías listas para usar")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🏗️ PASO 2 — Generar Dataset Sintético de Siniestros
# MAGIC
# MAGIC Vamos a crear **5,000 registros** que simulan datos reales de:
# MAGIC - Siniestros de seguros vehiculares en Perú
# MAGIC - Información de rutas, clima, tipo de vehículo, conductor
# MAGIC
# MAGIC **¿Por qué datos sintéticos?**
# MAGIC Porque cuando aprendes, no siempre tienes acceso a datos reales.
# MAGIC Los datos sintéticos te permiten practicar con estructuras realistas.

# COMMAND ----------

# Fijar semilla para reproducibilidad (siempre obtendrás los mismos datos)
np.random.seed(42)

N = 5000  # Número de registros

# --- CATÁLOGOS ---
regiones   = ["Lima", "Arequipa", "Cusco", "La Libertad", "Piura",
               "Cajamarca", "Junín", "Puno", "Loreto", "Ancash",
               "Ica", "Lambayeque", "Tacna", "Huánuco", "Apurímac"]

tipos_via  = ["Nacional", "Departamental", "Secundaria", "Urbana"]

tipos_veh  = ["Automóvil", "Camioneta", "Camión", "Bus", "Moto", "Semirremolque"]

causas     = ["Exceso de velocidad", "Falla mecánica", "Fatiga del conductor",
               "Lluvia / visibilidad reducida", "Invasión de carril",
               "Curva peligrosa", "Alcohol", "Distracción", "Neblina"]

franjas    = ["Madrugada (00-06)", "Mañana (06-12)", "Tarde (12-18)", "Noche (18-24)"]

# --- GENERAR COLUMNAS BASE ---
df_raw = pd.DataFrame({
    # Identificadores
    "id_siniestro"     : [f"SIN-2024-{str(i).zfill(5)}" for i in range(1, N+1)],

    # Temporal
    "fecha"            : pd.date_range(start="2022-01-01", periods=N, freq="8H").strftime("%Y-%m-%d"),
    "franja_horaria"   : np.random.choice(franjas, N, p=[0.15, 0.30, 0.35, 0.20]),
    "mes"              : np.random.randint(1, 13, N),

    # Ubicación
    "region"           : np.random.choice(regiones, N,
                             p=[0.25,0.08,0.07,0.07,0.07,0.06,0.06,0.06,0.05,0.05,
                                0.05,0.04,0.03,0.03,0.03]),
    "tipo_via"         : np.random.choice(tipos_via, N, p=[0.40, 0.25, 0.20, 0.15]),
    "km_ruta"          : np.round(np.random.uniform(1, 850, N), 1),

    # Vehículo
    "tipo_vehiculo"    : np.random.choice(tipos_veh, N,
                             p=[0.30, 0.20, 0.18, 0.12, 0.12, 0.08]),
    "antiguedad_veh"   : np.random.randint(0, 25, N),
    "cilindrada_cc"    : np.random.choice([1000, 1300, 1600, 2000, 2400, 3000, 4500], N),

    # Conductor
    "edad_conductor"   : np.random.randint(18, 75, N),
    "experiencia_anios": np.random.randint(0, 40, N),
    "infracciones_prev": np.random.poisson(lam=1.2, size=N),

    # Condición climática
    "lluvia_mm"        : np.round(np.abs(np.random.normal(25, 40, N)), 1),
    "visibilidad_km"   : np.round(np.clip(np.random.normal(12, 5, N), 0.5, 20), 1),
    "temperatura_c"    : np.round(np.random.normal(18, 8, N), 1),

    # Tráfico
    "flujo_vehicular"  : np.random.randint(50, 12000, N),
    "densidad_via"     : np.random.choice(["Baja", "Media", "Alta"], N, p=[0.30, 0.45, 0.25]),

    # Causa y resultado
    "causa_principal"  : np.random.choice(causas, N),
    "fallecidos"       : np.random.choice([0, 1, 2, 3, 4], N, p=[0.72, 0.15, 0.08, 0.03, 0.02]),
    "heridos"          : np.random.poisson(lam=2.1, size=N),
    "dano_vehiculo_sol": np.round(np.abs(np.random.lognormal(mean=9, sigma=1.2, size=N)), 0),
    "pago_seguro_sol"  : np.round(np.abs(np.random.lognormal(mean=9.5, sigma=1.0, size=N)), 0),
})

print(f"✅ Dataset generado: {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas")
print("\n📋 Primeras 3 filas:")
display(df_raw.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 PASO 3 — Crear la Variable Objetivo: Score de Riesgo
# MAGIC
# MAGIC Esta es la columna que el modelo intentará **predecir**.
# MAGIC La calculamos con una fórmula lógica que combina varios factores.
# MAGIC
# MAGIC **Niveles de riesgo:**
# MAGIC - 🔴 **ALTO** : Score > 66
# MAGIC - 🟡 **MEDIO**: Score 33–66
# MAGIC - 🟢 **BAJO** : Score < 33

# COMMAND ----------

def calcular_score_riesgo(row):
    """
    Calcula un score de riesgo entre 0 y 100.
    Cuanto mayor el score, mayor el riesgo de siniestro grave.
    """
    score = 0.0

    # Factor 1: Lluvia (hasta 20 puntos)
    score += min(row["lluvia_mm"] / 200 * 20, 20)

    # Factor 2: Visibilidad baja aumenta riesgo (hasta 15 puntos)
    score += max(0, (10 - row["visibilidad_km"]) / 10 * 15)

    # Factor 3: Flujo vehicular alto (hasta 15 puntos)
    score += min(row["flujo_vehicular"] / 12000 * 15, 15)

    # Factor 4: Antigüedad del vehículo (hasta 10 puntos)
    score += min(row["antiguedad_veh"] / 25 * 10, 10)

    # Factor 5: Infracciones previas (hasta 15 puntos)
    score += min(row["infracciones_prev"] * 4, 15)

    # Factor 6: Tipo de vía (rutas secundarias = más riesgo)
    score += {"Nacional": 5, "Departamental": 8, "Secundaria": 12, "Urbana": 6}[row["tipo_via"]]

    # Factor 7: Tipo de vehículo (vehículos pesados = más daño)
    score += {"Automóvil": 3, "Camioneta": 5, "Camión": 9,
              "Bus": 8, "Moto": 7, "Semirremolque": 10}[row["tipo_vehiculo"]]

    # Factor 8: Franja nocturna / madrugada
    if "Noche" in row["franja_horaria"] or "Madrugada" in row["franja_horaria"]:
        score += 8

    # Normalizar a 0–100 con algo de ruido aleatorio
    score = score + np.random.normal(0, 3)
    return round(float(np.clip(score, 0, 100)), 2)

# Aplicar la función fila por fila
df_raw["score_riesgo"] = df_raw.apply(calcular_score_riesgo, axis=1)

# Crear etiqueta categórica
df_raw["nivel_riesgo"] = pd.cut(
    df_raw["score_riesgo"],
    bins=[0, 33, 66, 100],
    labels=["BAJO", "MEDIO", "ALTO"]
)

print("📊 Distribución de niveles de riesgo:")
print(df_raw["nivel_riesgo"].value_counts())
print(f"\nScore promedio: {df_raw['score_riesgo'].mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💉 PASO 4 — Introducir Errores Realistas (para practicar limpieza)
# MAGIC
# MAGIC En el mundo real los datos SIEMPRE tienen problemas.
# MAGIC Vamos a introducir estos errores intencionalmente para luego limpiarlos:
# MAGIC - **Valores nulos** (datos que faltan)
# MAGIC - **Outliers** (valores absurdos)
# MAGIC - **Errores de formato** (fechas mal escritas, espacios extra)
# MAGIC - **Duplicados**

# COMMAND ----------

df_dirty = df_raw.copy()

# --- 4a. Introducir NULOS (aprox 8% de datos faltantes en varias columnas) ---
cols_con_nulos = ["lluvia_mm", "visibilidad_km", "flujo_vehicular",
                   "edad_conductor", "experiencia_anios", "temperatura_c"]
for col in cols_con_nulos:
    mask = np.random.choice([True, False], size=N, p=[0.08, 0.92])
    df_dirty.loc[mask, col] = np.nan

# --- 4b. Introducir OUTLIERS absurdos ---
idx_outliers = np.random.choice(N, size=80, replace=False)
df_dirty.loc[idx_outliers[:20], "edad_conductor"]    = np.random.choice([5, 150, 200, -1], 20)
df_dirty.loc[idx_outliers[20:40], "lluvia_mm"]       = np.random.choice([9999, -50, 5000], 20)
df_dirty.loc[idx_outliers[40:60], "flujo_vehicular"] = np.random.choice([-100, 99999, 0], 20)
df_dirty.loc[idx_outliers[60:],   "pago_seguro_sol"] = -1

# --- 4c. Introducir errores de FORMATO en fechas ---
idx_fechas = np.random.choice(N, size=50, replace=False)
fechas_malas = ["31/13/2024", "2024-99-01", "no-date", "  2024/01/01  "]
df_dirty.loc[idx_fechas, "fecha"] = np.random.choice(fechas_malas, 50)

# --- 4d. Introducir DUPLICADOS (50 filas duplicadas) ---
filas_dup = df_dirty.sample(50, random_state=7)
df_dirty = pd.concat([df_dirty, filas_dup], ignore_index=True)

print(f"📦 Dataset con errores: {df_dirty.shape[0]:,} filas")
print(f"🔴 Nulos por columna:\n{df_dirty[cols_con_nulos].isnull().sum()}")
print(f"🔴 Duplicados: {df_dirty.duplicated().sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🧹 PASO 5 — Feature Engineering: Limpiar los Datos
# MAGIC
# MAGIC Ahora viene el trabajo real. Vamos paso a paso:
# MAGIC
# MAGIC ### 5.1 — Eliminar duplicados

# COMMAND ----------

df_clean = df_dirty.copy()

filas_antes = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=["id_siniestro"], keep="first")
print(f"✅ Duplicados eliminados: {filas_antes - len(df_clean)}")
print(f"   Filas restantes: {len(df_clean):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 — Limpiar y corregir fechas

# COMMAND ----------

def limpiar_fecha(fecha_str):
    """Intenta parsear una fecha en múltiples formatos. Si falla, devuelve NaT."""
    formatos = ["%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"]
    fecha_str = str(fecha_str).strip()
    for fmt in formatos:
        try:
            return pd.to_datetime(fecha_str, format=fmt)
        except:
            continue
    return pd.NaT  # No se pudo parsear

df_clean["fecha"] = df_clean["fecha"].apply(limpiar_fecha)

fechas_invalidas = df_clean["fecha"].isnull().sum()
print(f"✅ Fechas inválidas detectadas y marcadas como nulas: {fechas_invalidas}")

# Rellenar fechas inválidas con la fecha mediana
fecha_mediana = df_clean["fecha"].median()
df_clean["fecha"] = df_clean["fecha"].fillna(fecha_mediana)
print(f"✅ Fechas nulas rellenadas con: {fecha_mediana.date()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 — Tratar Outliers con reglas de negocio

# COMMAND ----------

# Regla: la edad del conductor debe estar entre 18 y 85 años
mask_edad = (df_clean["edad_conductor"] < 18) | (df_clean["edad_conductor"] > 85)
print(f"🔍 Conductores con edad inválida: {mask_edad.sum()}")
df_clean.loc[mask_edad, "edad_conductor"] = df_clean["edad_conductor"].median()

# Regla: lluvia no puede ser negativa ni mayor a 500 mm/día
mask_lluvia = (df_clean["lluvia_mm"] < 0) | (df_clean["lluvia_mm"] > 500)
print(f"🔍 Registros con lluvia inválida: {mask_lluvia.sum()}")
df_clean.loc[mask_lluvia, "lluvia_mm"] = df_clean["lluvia_mm"].median()

# Regla: flujo vehicular entre 0 y 50,000
mask_flujo = (df_clean["flujo_vehicular"] < 0) | (df_clean["flujo_vehicular"] > 50000)
print(f"🔍 Registros con flujo inválido: {mask_flujo.sum()}")
df_clean.loc[mask_flujo, "flujo_vehicular"] = df_clean["flujo_vehicular"].median()

# Regla: pagos no pueden ser negativos
mask_pago = df_clean["pago_seguro_sol"] < 0
print(f"🔍 Pagos negativos: {mask_pago.sum()}")
df_clean.loc[mask_pago, "pago_seguro_sol"] = 0

print("\n✅ Outliers corregidos con valores medianos")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 — Imputar valores nulos

# COMMAND ----------

# Columnas numéricas: rellenar con la MEDIANA (más robusta que la media)
cols_numericas = ["lluvia_mm", "visibilidad_km", "flujo_vehicular",
                   "edad_conductor", "experiencia_anios", "temperatura_c"]

for col in cols_numericas:
    nulos_antes = df_clean[col].isnull().sum()
    mediana = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(mediana)
    print(f"  ✅ {col:25s} → {nulos_antes} nulos → rellenados con mediana={mediana:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 — Crear nuevas variables (Feature Engineering)
# MAGIC
# MAGIC Aquí **creamos columnas nuevas** a partir de las existentes.
# MAGIC Esto ayuda al modelo a aprender mejor los patrones.

# COMMAND ----------

# Variable 1: ¿Es horario de riesgo? (noche o madrugada)
df_clean["es_horario_riesgo"] = df_clean["franja_horaria"].apply(
    lambda x: 1 if "Noche" in x or "Madrugada" in x else 0
)

# Variable 2: Índice de peligrosidad climática
df_clean["indice_clima_peligro"] = (
    (df_clean["lluvia_mm"] / df_clean["lluvia_mm"].max()) * 0.5 +
    (1 - df_clean["visibilidad_km"] / df_clean["visibilidad_km"].max()) * 0.5
).round(4)

# Variable 3: Ratio experiencia / edad (conductor maduro y experimentado = menor riesgo)
df_clean["ratio_exp_edad"] = (
    df_clean["experiencia_anios"] / df_clean["edad_conductor"]
).round(4)

# Variable 4: ¿Vehículo antiguo? (más de 10 años)
df_clean["vehiculo_antiguo"] = (df_clean["antiguedad_veh"] > 10).astype(int)

# Variable 5: Estación del año según mes
df_clean["estacion"] = df_clean["mes"].map({
    12: "Verano", 1: "Verano", 2: "Verano",
    3: "Otoño",  4: "Otoño",  5: "Otoño",
    6: "Invierno", 7: "Invierno", 8: "Invierno",
    9: "Primavera", 10: "Primavera", 11: "Primavera"
})

print(f"✅ Nuevas columnas creadas: es_horario_riesgo, indice_clima_peligro,")
print(f"                           ratio_exp_edad, vehiculo_antiguo, estacion")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.6 — Convertir variables categóricas a numéricas
# MAGIC
# MAGIC Los modelos de ML solo entienden números. Necesitamos convertir:
# MAGIC - "Camión" → número
# MAGIC - "Nacional" → número

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Columnas categóricas que usará el modelo
cols_categoricas = ["tipo_via", "tipo_vehiculo", "densidad_via",
                     "region", "estacion", "franja_horaria"]

for col in cols_categoricas:
    df_clean[f"{col}_cod"] = le.fit_transform(df_clean[col].astype(str))
    print(f"  ✅ {col} → {col}_cod   (valores: {df_clean[col].unique()[:3]}...)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📊 PASO 6 — Resumen del dataset limpio

# COMMAND ----------

print("=" * 55)
print("📋 RESUMEN FINAL DEL DATASET LIMPIO")
print("=" * 55)
print(f"  Filas totales         : {len(df_clean):,}")
print(f"  Columnas              : {df_clean.shape[1]}")
print(f"  Nulos restantes       : {df_clean.isnull().sum().sum()}")
print(f"  Score promedio        : {df_clean['score_riesgo'].mean():.2f}")
print(f"  % Nivel ALTO          : {(df_clean['nivel_riesgo']=='ALTO').mean()*100:.1f}%")
print(f"  % Nivel MEDIO         : {(df_clean['nivel_riesgo']=='MEDIO').mean()*100:.1f}%")
print(f"  % Nivel BAJO          : {(df_clean['nivel_riesgo']=='BAJO').mean()*100:.1f}%")
print("=" * 55)

display(df_clean.describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 💾 PASO 7 — Guardar en Databricks (Delta Table)
# MAGIC
# MAGIC Guardamos el dataset limpio para que el Notebook 2 (Modelo ML) lo pueda usar.

# COMMAND ----------

# Convertir a Spark DataFrame para guardar en Delta
#spark_df = spark.createDataFrame(df_clean)

# Guardar como tabla Delta en el catálogo de Databricks
#spark_df.write.format("delta").mode("overwrite").saveAsTable("siniestros_limpios")

#print("✅ Tabla guardada: siniestros_limpios")
#print(f"   Filas: {spark_df.count():,}")

# También guardamos como CSV por si quieres descargarlo
#df_clean.to_csv("/dbfs/FileStore/siniestros_limpios.csv", index=False)
#print("✅ CSV guardado en: /dbfs/FileStore/siniestros_limpios.csv")

# PASO 7 CORREGIDO — Guardar en Databricks Serverless
# -------------------------------------------------------
# En Serverless NO existe /dbfs/FileStore/
# Usamos Unity Catalog o Volumes en su lugar

# ✅ OPCIÓN 1: Guardar como Delta Table (siempre funciona)
spark_df = spark.createDataFrame(df_clean)
spark_df.write.format("delta").mode("overwrite").saveAsTable("siniestros_limpios")
print(f"✅ Tabla guardada: siniestros_limpios")
print(f"   Filas: {spark_df.count():,}")

# ✅ OPCIÓN 2: Guardar CSV en Volumes (Serverless sí soporta esto)
try:
    # Crear el Volume si no existe (solo la primera vez)
    spark.sql("CREATE SCHEMA IF NOT EXISTS main.semaforo")
    spark.sql("CREATE VOLUME IF NOT EXISTS main.semaforo.archivos")
    
    df_clean.to_csv(
        "/Volumes/main/semaforo/archivos/siniestros_limpios.csv",
        index=False
    )
    print("✅ CSV guardado en: /Volumes/main/semaforo/archivos/siniestros_limpios.csv")
    csv_guardado = True

except Exception as e:
    print(f"⚠️ No se pudo guardar el CSV en Volumes: {e}")
    print("   El CSV no es crítico — la Delta Table es suficiente para continuar.")
    csv_guardado = False

print("\n🎯 LISTO: El Notebook 2 cargará los datos desde la Delta Table 'siniestros_limpios'")



# COMMAND ----------

# MAGIC %md
# MAGIC ## ✅ ¡NOTEBOOK 1 COMPLETADO!
# MAGIC
# MAGIC **Lo que hiciste:**
# MAGIC - ✅ Generaste 5,000 registros sintéticos de siniestros viales
# MAGIC - ✅ Eliminaste duplicados (50 registros)
# MAGIC - ✅ Corregiste fechas inválidas
# MAGIC - ✅ Trataste outliers con reglas de negocio
# MAGIC - ✅ Imputaste valores nulos con la mediana
# MAGIC - ✅ Creaste 5 nuevas variables (Feature Engineering)
# MAGIC - ✅ Codificaste variables categóricas
# MAGIC - ✅ Guardaste en Delta Table y CSV
# MAGIC
# MAGIC **➡️ Siguiente paso: Abre el NOTEBOOK_2_Modelo_ML.py**