"""Guía visual y simulador educativo de los modelos entrenados."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
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
MODEL_EXPLANATIONS = {
    "CART": {
        "family": "Árbol de decisión · CART",
        "summary": (
            "Construye un árbol binario de reglas. En cada paso elige una variable "
            "y un umbral que separen mejor las clases; al llegar a una hoja, predice "
            "a partir de los ejemplos que quedaron allí."
        ),
        "math": r"Gini(D)=1-\sum_k p_k^2",
        "learning": (
            "El algoritmo prueba divisiones y escoge la que más reduce la impureza "
            "ponderada de sus dos grupos. Es una búsqueda voraz: mejora el nodo actual, "
            "no busca todos los árboles posibles. La profundidad, el tamaño mínimo de "
            "las hojas y la poda ayudan a evitar un árbol demasiado complejo."
        ),
        "strength": "Es el más fácil de dibujar, explicar y convertir en reglas.",
        "tradeoff": (
            "Un pequeño cambio en los datos puede cambiar sus divisiones; sin control "
            "de complejidad puede sobreajustar."
        ),
        "use": (
            "Reglas de decisión auditables, evaluación inicial de variables y situaciones "
            "en que explicar una ruta de decisión importa."
        ),
    },
    "Random Forest": {
        "family": "Ensamble · Bagging",
        "summary": (
            "Entrena muchos árboles con variaciones aleatorias de los datos y de las "
            "variables disponibles en cada división. Combina sus probabilidades para "
            "producir una predicción más estable que la de un árbol individual."
        ),
        "math": r"\hat p(y=1\mid x)=\frac{1}{T}\sum_{t=1}^{T}\hat p_t(y=1\mid x)",
        "learning": (
            "Cada árbol se ajusta por separado; en este proyecto se exploran el número "
            "de árboles, las variables candidatas por división, la profundidad, el "
            "muestreo bootstrap y el tamaño mínimo de las hojas."
        ),
        "strength": (
            "El promedio suele reducir la variabilidad de un árbol y capta relaciones "
            "no lineales sin exigir escalar las variables."
        ),
        "tradeoff": (
            "Es menos fácil de interpretar como un conjunto que un árbol; guardar y "
            "predecir con muchos árboles puede requerir más tiempo y memoria."
        ),
        "use": (
            "Clasificación tabular cuando importa un rendimiento robusto y hay "
            "interacciones entre variables."
        ),
    },
    "XGBoost": {
        "family": "Ensamble · Gradient Boosting",
        "summary": (
            "Agrega árboles pequeños uno tras otro. Cada árbol nuevo intenta mejorar "
            "los errores de las predicciones acumuladas; una tasa de aprendizaje "
            "controla cuánto aporta cada árbol."
        ),
        "math": (
            r"F_m(x)=F_{m-1}(x)+\eta f_m(x)"
            r"\qquad \hat p(y=1\mid x)=\frac{1}{1+e^{-F_m(x)}}"
        ),
        "learning": (
            "Optimiza una pérdida logística más penalizaciones que controlan la "
            "complejidad de los árboles. XGBoost usa gradientes y segundas derivadas "
            "para escoger divisiones; aquí se ajustan la cantidad y profundidad de "
            "árboles, la tasa de aprendizaje, el submuestreo y la regularización."
        ),
        "strength": (
            "Puede aprender patrones complejos y suele rendir muy bien en datos "
            "tabulares."
        ),
        "tradeoff": (
            "Los árboles son secuenciales, hay más hiperparámetros y cuesta más explicar "
            "la predicción global que con un único árbol."
        ),
        "use": (
            "Clasificación tabular con relaciones no lineales, cuando es posible "
            "comparar y ajustar cuidadosamente varias configuraciones."
        ),
    },
}

st.set_page_config(
    page_title="Árboles y ensambles | Equipo 1",
    page_icon="🌳",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
    .hero {
        padding: 2rem 2.2rem; border-radius: 1.2rem; margin-bottom: 1.2rem;
        background: linear-gradient(120deg, #123b55 0%, #176b70 58%, #68a88b 100%);
        color: white;
    }
    .hero h1 {color: white; margin-bottom: .35rem; font-size: 2.5rem;}
    .hero p {color: #e7f4f0; font-size: 1.08rem; margin-bottom: 0;}
    .note {
        padding: 1rem 1.2rem; border-left: 5px solid #d59b27;
        border-radius: .35rem; background: #fff8e9; margin: .7rem 0 1rem;
    }
    .small-label {
        color: #438175; text-transform: uppercase; letter-spacing: .08em;
        font-size: .78rem; font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="small-label">Inteligencia Artificial · Equipo 1</div>
      <h1>Árboles de decisión y ensambles</h1>
      <p>Una guía para entender, comparar y probar CART, Random Forest y XGBoost.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "**Demostración académica, no clínica.** El conjunto es pequeño y de referencia; "
    "sus resultados no validan una herramienta médica ni sirven para diagnosticar "
    "o tomar decisiones sobre pacientes."
)

metrics_path = RESULTS_DIR / "metricas.csv"
params_path = RESULTS_DIR / "mejores_parametros.json"
metadata_path = RESULTS_DIR / "metadata.json"
examples_path = DATA_DIR / "ejemplos_prueba.csv"
missing_artifacts = [
    str(path.relative_to(ROOT))
    for path in [*MODELS.values(), metrics_path, params_path, metadata_path, examples_path]
    if not path.exists()
]
if missing_artifacts:
    st.error(
        "Faltan archivos generados por el experimento: "
        + ", ".join(missing_artifacts)
        + ". Ejecuta `python src\\experiment.py` desde la carpeta del proyecto."
    )
    st.stop()

metrics = pd.read_csv(metrics_path)
best_params = json.loads(params_path.read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
examples = pd.read_csv(examples_path)

overview, workflow, models_tab, results_tab, simulator = st.tabs(
    ["🎯 Objetivo", "🧭 Cómo se hizo", "🌳 Los modelos", "📊 Resultados", "🧪 Simulador"]
)

with overview:
    st.header("¿Qué problema resuelve este proyecto?")
    st.write(
        "Es un problema de **clasificación supervisada**: a partir de 30 mediciones "
        "numéricas de una muestra, un modelo aprende con ejemplos ya etiquetados y "
        "predice una de dos clases: **benigno** o **maligno**. Comparamos tres formas "
        "de construir modelos basados en árboles y observamos en qué se parecen y "
        "en qué se diferencian."
    )
    left, right, third = st.columns(3)
    left.metric("Muestras", f"{metadata['n_samples']:,}")
    right.metric("Mediciones por muestra", metadata["n_features"])
    third.metric("Modelos comparados", len(MODELS))

    st.subheader("¿De dónde vienen los datos?")
    st.write(
        "Usamos Breast Cancer Wisconsin (Diagnostic), una colección de referencia "
        "distribuida por scikit-learn y procedente de UCI. Sus 569 filas contienen "
        "mediciones numéricas de características de núcleos celulares obtenidas de "
        "imágenes digitalizadas de aspiraciones con aguja fina. No son imágenes que "
        "la app pueda analizar: recibe las mediciones tabulares ya calculadas."
    )
    benign, malignant = st.columns(2)
    benign.metric("Casos benignos", metadata["class_counts"]["benigno"])
    malignant.metric("Casos malignos", metadata["class_counts"]["maligno"])
    st.caption(
        "En la etiqueta usada aquí, 0 = benigno y 1 = maligno. La clase positiva "
        "se fijó como maligno para interpretar directamente su recall."
    )

    with st.expander("¿Qué significan las columnas de entrada?"):
        st.write(
            "Son 10 características celulares calculadas en tres formas: promedio "
            "(`mean`), variación (`error`) y peor valor (`worst`). Por ejemplo, "
            "`mean radius` es el radio promedio y `worst area` es el área del grupo "
            "de valores más altos. Todas deben ser números."
        )
        st.code(
            "mean radius, mean texture, mean perimeter, ... , worst fractal dimension",
            language="text",
        )

with workflow:
    st.header("De los datos a una predicción")
    st.write(
        "El objetivo experimental es comparar los algoritmos con las mismas filas "
        "reservadas, para que la comparación sea justa."
    )
    stages = st.columns(4)
    stage_content = [
        ("1 · Preparar", "Separar variables (X) de la etiqueta (y). No se escalan: los árboles dividen por umbrales."),
        ("2 · Separar", "455 filas para entrenar y 114 para prueba. La división estratificada conserva la proporción de clases."),
        ("3 · Ajustar", "20 configuraciones aleatorias por modelo y validación cruzada estratificada de 5 pliegues."),
        ("4 · Evaluar", "Elegir parámetros por ROC-AUC promedio y medir una vez en la prueba, que no participó en el ajuste."),
    ]
    for col, (title, description) in zip(stages, stage_content):
        with col:
            st.markdown(f"#### {title}")
            st.write(description)

    st.subheader("¿Qué significa validación cruzada de 5 pliegues?")
    st.write(
        "Durante la búsqueda, los datos de entrenamiento se dividen en cinco partes. "
        "Se entrena con cuatro y se valida con la restante; el proceso rota para que "
        "cada parte se valide una vez. Se promedia ROC-AUC para escoger la "
        "configuración. Después, el conjunto de prueba reservado se usa para la "
        "comparación final."
    )
    st.info(
        "**Semilla reproducible:** 42. **Clase positiva:** maligno. "
        "**Umbral de clase:** 0.5. No se optimizó un umbral para uso clínico."
    )

with models_tab:
    st.header("Tres modelos, tres estrategias")
    st.write(
        "Todos aprenden reglas basadas en árboles, pero la manera de combinar esas "
        "reglas cambia su estabilidad, complejidad y forma de aprender."
    )
    for model_name, details in MODEL_EXPLANATIONS.items():
        with st.expander(f"{model_name} · {details['family']}", expanded=True):
            st.write(details["summary"])
            st.latex(details["math"])
            st.markdown(f"**¿Cómo aprende?** {details['learning']}")
            advantages, limits = st.columns(2)
            with advantages:
                st.markdown(f"**Ventaja:** {details['strength']}")
            with limits:
                st.markdown(f"**Límite:** {details['tradeoff']}")
            st.markdown(f"**¿Dónde puede ser útil?** {details['use']}")
            st.caption(
                "La aplicación estima asociaciones a partir de ejemplos etiquetados; "
                "no establece causas ni sustituye conocimiento experto."
            )
    st.subheader("Comparación rápida")
    st.dataframe(
        pd.DataFrame(
            [
                ["CART", "Un árbol", "Voraz: mejora una división cada vez", "Alta"],
                ["Random Forest", "Promedio de muchos árboles", "Bagging y aleatoriedad", "Media"],
                ["XGBoost", "Suma secuencial de árboles", "Boosting con pérdida y regularización", "Menor"],
            ],
            columns=["Modelo", "Qué combina", "Estrategia", "Interpretabilidad global"],
        ),
        hide_index=True,
        use_container_width=True,
    )

with results_tab:
    st.header("¿Cómo se midió el rendimiento?")
    st.write(
        "Los valores siguientes corresponden únicamente a 114 filas de prueba "
        "reservadas. Las métricas se leen para la clase positiva **maligno**; un "
        "resultado alto aquí no garantiza el mismo rendimiento en otros datos."
    )
    metric_labels = {
        "roc_auc": "ROC-AUC",
        "precision_maligno": "Precisión · maligno",
        "recall_maligno": "Recall · maligno",
        "f1_maligno": "F1 · maligno",
        "exactitud": "Exactitud",
    }
    chart_data = metrics.set_index("modelo")[list(metric_labels)].rename(
        columns=metric_labels
    )
    st.bar_chart(chart_data, y=list(metric_labels.values()))
    st.dataframe(
        metrics[
            [
                "modelo",
                "roc_auc",
                "precision_maligno",
                "recall_maligno",
                "f1_maligno",
                "exactitud",
                "verdaderos_negativos",
                "falsos_positivos",
                "falsos_negativos",
                "verdaderos_positivos",
            ]
        ].rename(
            columns={
                "modelo": "Modelo",
                "roc_auc": "ROC-AUC",
                "precision_maligno": "Precisión (maligno)",
                "recall_maligno": "Recall (maligno)",
                "f1_maligno": "F1 (maligno)",
                "exactitud": "Exactitud",
                "verdaderos_negativos": "TN",
                "falsos_positivos": "FP",
                "falsos_negativos": "FN",
                "verdaderos_positivos": "TP",
            }
        ).round(3),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("¿Cómo explico estas métricas?"):
        st.markdown(
            "- **Precisión:** de lo que el modelo llamó maligno, ¿qué proporción era maligno?\n"
            "- **Recall o sensibilidad:** de los casos malignos reales, ¿cuántos detectó?\n"
            "- **F1:** equilibrio entre precisión y recall.\n"
            "- **ROC-AUC:** capacidad de ordenar casos positivos por encima de negativos "
            "al recorrer distintos umbrales; no es una exactitud.\n"
            "- **Matriz de confusión:** TP = maligno detectado, FN = maligno no detectado, "
            "FP = benigno marcado como maligno, TN = benigno correctamente reconocido."
        )

    st.subheader("Visualizaciones del experimento")
    figures = [
        ("comparacion_metricas.png", "Comparación de métricas"),
        ("curvas_roc.png", "Curvas ROC"),
        ("matrices_confusion.png", "Matrices de confusión"),
        ("importancia_variables.png", "Importancia de variables"),
        ("arbol_cart.png", "Vista parcial del árbol CART"),
    ]
    for offset in range(0, len(figures), 2):
        first, second = st.columns(2)
        for col, (filename, caption) in zip((first, second), figures[offset : offset + 2]):
            with col:
                figure_path = RESULTS_DIR / filename
                if figure_path.exists():
                    st.image(str(figure_path), caption=caption, use_container_width=True)
    st.caption(
        "La importancia es una señal interna del modelo, no una prueba de causalidad. "
        "Las curvas ROC y los gráficos resumen este conjunto de prueba."
    )

    st.subheader("Configuración elegida")
    st.write(
        "La tabla muestra los hiperparámetros de la mejor configuración encontrada "
        "por RandomizedSearchCV en cada modelo. No son reglas universales: cambian "
        "según datos, partición y búsqueda."
    )
    st.json(best_params)

with simulator:
    st.header("Prueba los modelos")
    st.write(
        "Escoge un modelo y examina un ejemplo reservado o carga mediciones en CSV. "
        "El modelo devuelve una clase con umbral 0.5 y una probabilidad estimada "
        "para la clase 1 (maligno)."
    )
    model_name = st.selectbox("Modelo para simular", list(MODELS))
    model = joblib.load(MODELS[model_name])
    expected = list(model.feature_names_in_)
    input_mode = st.radio(
        "Origen de las mediciones",
        ["Ejemplo reservado de prueba", "Cargar archivo CSV"],
        horizontal=True,
    )

    if input_mode == "Ejemplo reservado de prueba":
        selected = st.number_input(
            "Número de ejemplo",
            min_value=1,
            max_value=len(examples),
            value=1,
            step=1,
        )
        sample = examples.iloc[[int(selected) - 1]]
        st.caption(
            "Son filas reservadas para prueba; aquí no se muestra su etiqueta real "
            "para que puedas observar la predicción del modelo."
        )
        st.dataframe(sample, hide_index=True, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="note"><strong>Qué subir:</strong> un archivo .csv con una fila
            por muestra y las 30 columnas numéricas del conjunto. Los nombres de las
            columnas deben coincidir con la plantilla; no incluyas una columna
            <code>maligno</code> ni identificadores. No se suben imágenes.</div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Descargar plantilla CSV",
            data=examples.head(1).to_csv(index=False).encode("utf-8"),
            file_name="plantilla_entrada.csv",
            mime="text/csv",
        )
        uploaded = st.file_uploader("Selecciona tu CSV", type="csv")
        if uploaded is None:
            st.info("Sube un CSV para ver una predicción.")
            st.stop()
        try:
            sample = pd.read_csv(uploaded)
        except (pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"No se pudo leer el CSV. Comprueba su formato y codificación: {error}")
            st.stop()
        if sample.empty:
            st.error("El CSV no contiene filas de datos.")
            st.stop()

    missing = [column for column in expected if column not in sample.columns]
    if missing:
        st.error(f"Faltan columnas requeridas: {', '.join(missing)}")
        st.stop()
    sample = sample.loc[:, expected]
    non_numeric = [
        column
        for column in expected
        if not pd.api.types.is_numeric_dtype(sample[column])
    ]
    if non_numeric:
        st.error(
            "Estas columnas deben contener solo números: " + ", ".join(non_numeric)
        )
        st.stop()
    if sample.isna().any().any():
        st.error("Hay celdas vacías. Completa los valores antes de predecir.")
        st.stop()
    if not np.isfinite(sample.to_numpy(dtype=float)).all():
        st.error("Todas las mediciones deben ser valores numéricos finitos.")
        st.stop()

    probabilities = model.predict_proba(sample)[:, 1]
    predictions = model.predict(sample)
    prediction_results = pd.DataFrame(
        {
            "Predicción": [
                "Maligno" if label == 1 else "Benigno" for label in predictions
            ],
            "Probabilidad estimada de malignidad": probabilities,
        },
        index=sample.index,
    )
    st.subheader("Resultado de la simulación")
    st.dataframe(prediction_results, hide_index=True, use_container_width=True)
    st.caption(
        "La probabilidad es una salida de este modelo y estos datos; no equivale a "
        "una probabilidad clínica calibrada."
    )
