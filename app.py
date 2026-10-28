"""Simulador educativo de los modelos entrenados."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
MODELS = {
    "CART": RESULTS_DIR / "cart.joblib",
    "Random Forest": RESULTS_DIR / "random_forest.joblib",
    "XGBoost": RESULTS_DIR / "xgboost.joblib",
}

st.set_page_config(page_title="Simulador de árboles", page_icon="🌳", layout="wide")
st.title("Simulación interactiva: modelos de árboles")
st.warning(
    "Demostración académica con datos de referencia. No es una herramienta "
    "médica y no debe usarse para diagnóstico ni decisiones clínicas."
)
st.write(
    "Seleccione un modelo y evalúe registros de prueba reservados o cargue "
    "un CSV con las 30 variables numéricas del conjunto."
)

missing_models = [name for name, path in MODELS.items() if not path.exists()]
examples_path = DATA_DIR / "ejemplos_prueba.csv"
if missing_models or not examples_path.exists():
    st.error(
        "Faltan modelos o ejemplos. Ejecute primero "
        "`python src\\experiment.py` desde la carpeta del proyecto."
    )
    st.stop()

model_name = st.selectbox("Modelo", list(MODELS))
model = joblib.load(MODELS[model_name])
examples = pd.read_csv(examples_path)
input_mode = st.radio("Entrada", ["Ejemplo reservado de prueba", "Cargar archivo CSV"])

if input_mode == "Ejemplo reservado de prueba":
    selected = st.number_input(
        "Número de ejemplo", min_value=1, max_value=len(examples), value=1, step=1
    )
    sample = examples.iloc[[int(selected) - 1]]
    st.dataframe(sample, use_container_width=True)
else:
    uploaded = st.file_uploader("CSV con una o más filas y 30 variables", type="csv")
    if uploaded is None:
        st.info("Suba un archivo CSV para obtener predicciones.")
        st.stop()
    sample = pd.read_csv(uploaded)

expected = list(model.feature_names_in_)
missing = [column for column in expected if column not in sample.columns]
if missing:
    st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
    st.stop()
sample = sample.loc[:, expected]
if not all(pd.api.types.is_numeric_dtype(sample[column]) for column in expected):
    st.error("Todas las variables de entrada deben ser numéricas.")
    st.stop()
if sample.isna().any().any():
    st.error("La entrada contiene valores vacíos; complételos antes de predecir.")
    st.stop()

probabilities = model.predict_proba(sample)[:, 1]
predictions = model.predict(sample)
results = pd.DataFrame(
    {
        "Predicción": ["Maligno" if label == 1 else "Benigno" for label in predictions],
        "Probabilidad estimada de malignidad": probabilities,
    },
    index=sample.index,
)
st.subheader("Resultado de la simulación")
st.dataframe(results, use_container_width=True)
st.caption(
    "Las probabilidades son salidas del modelo en este conjunto; no equivalen "
    "a una probabilidad clínica calibrada."
)
