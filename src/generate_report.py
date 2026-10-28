"""Genera el informe PDF a partir de los resultados guardados por experiment.py."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "informe_tecnico.pdf"
ACCENT = colors.HexColor("#174A6E")
PALE_BLUE = colors.HexColor("#EAF2F8")
MID_GRAY = colors.HexColor("#555555")


def register_fonts() -> None:
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    regular = font_dir / "DejaVuSans.ttf"
    bold = font_dir / "DejaVuSans-Bold.ttf"
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("DV", str(regular)))
        pdfmetrics.registerFont(TTFont("DV-Bold", str(bold)))


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_table(rows: list[list], widths: list[float], header: bool = True) -> Table:
    wrapped = []
    styles = getSampleStyleSheet()
    for row_index, row in enumerate(rows):
        wrapped_row = []
        for value in row:
            if isinstance(value, Paragraph):
                wrapped_row.append(value)
            else:
                wrapped_row.append(
                    Paragraph(str(value), styles["BodyText"] if row_index else styles["BodyText"])
                )
        wrapped.append(wrapped_row)
    table = Table(wrapped, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB7C4")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "DV-Bold"),
            ]
        )
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE_BLUE))
    table.setStyle(TableStyle(commands))
    return table


def add_image(story: list, filename: str, width: float, height: float) -> None:
    path = RESULTS / filename
    if not path.exists():
        story.append(paragraph(f"No se encontró la figura {filename}.", STYLES["Small"]))
        return
    image = Image(str(path))
    scale = min(width / image.imageWidth, height / image.imageHeight, 1)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    story.extend([image, Spacer(1, 0.25 * cm)])


def fmt(value: float) -> str:
    return f"{value:.3f}"


def build_report() -> None:
    metrics_path = RESULTS / "metricas.csv"
    metadata_path = RESULTS / "metadata.json"
    params_path = RESULTS / "mejores_parametros.json"
    if not all(path.exists() for path in (metrics_path, metadata_path, params_path)):
        raise FileNotFoundError(
            "Faltan resultados. Ejecute primero `python src\\experiment.py`."
        )

    metrics = pd.read_csv(metrics_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    params = json.loads(params_path.read_text(encoding="utf-8"))
    ranked = metrics.sort_values("roc_auc", ascending=False)
    best = ranked.iloc[0]
    fonts = "DV" if "DV" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    bold = "DV-Bold" if "DV-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"

    global STYLES
    base = getSampleStyleSheet()
    STYLES = {
        "Title": ParagraphStyle(
            "TitleCustom", parent=base["Title"], fontName=bold, fontSize=22,
            leading=28, textColor=ACCENT, alignment=TA_CENTER, spaceAfter=12,
        ),
        "Subtitle": ParagraphStyle(
            "SubtitleCustom", parent=base["Normal"], fontName=fonts, fontSize=12,
            leading=18, textColor=MID_GRAY, alignment=TA_CENTER, spaceAfter=8,
        ),
        "Heading1": ParagraphStyle(
            "Heading1Custom", parent=base["Heading1"], fontName=bold, fontSize=15,
            leading=19, textColor=ACCENT, spaceBefore=13, spaceAfter=7,
        ),
        "Heading2": ParagraphStyle(
            "Heading2Custom", parent=base["Heading2"], fontName=bold, fontSize=11,
            leading=14, textColor=ACCENT, spaceBefore=8, spaceAfter=4,
        ),
        "Body": ParagraphStyle(
            "BodyCustom", parent=base["BodyText"], fontName=fonts, fontSize=9.2,
            leading=13.3, alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        "Formula": ParagraphStyle(
            "FormulaCustom", parent=base["BodyText"], fontName=fonts, fontSize=9.2,
            leading=14, alignment=TA_CENTER, backColor=PALE_BLUE,
            borderPadding=7, spaceBefore=3, spaceAfter=8,
        ),
        "Small": ParagraphStyle(
            "SmallCustom", parent=base["BodyText"], fontName=fonts, fontSize=7.6,
            leading=10, textColor=MID_GRAY, spaceAfter=4,
        ),
        "Table": ParagraphStyle(
            "TableCustom", parent=base["BodyText"], fontName=fonts, fontSize=7.3,
            leading=9.5,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeaderCustom", parent=base["BodyText"], fontName=bold,
            fontSize=7.3, leading=9.5, textColor=colors.white,
        ),
    }

    def P(text: str, style: str = "Body") -> Paragraph:
        return paragraph(text, STYLES[style])

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
        title="Comparación de CART, Random Forest y XGBoost",
        author="Equipo 1 — Inteligencia Artificial",
        subject="Informe técnico académico de aprendizaje supervisado",
    )
    story: list = [
        Spacer(1, 2.4 * cm),
        P("ACTIVIDAD PRÁCTICA · INTELIGENCIA ARTIFICIAL", "Subtitle"),
        Spacer(1, 0.4 * cm),
        P("Árboles de decisión y métodos de ensamble", "Title"),
        P("Comparación experimental de CART, Random Forest y XGBoost", "Subtitle"),
        Spacer(1, 0.5 * cm),
        P("Equipo 1", "Subtitle"),
        P("Informe técnico · Modelado matemático · Simulación · Análisis de aplicaciones", "Subtitle"),
        Spacer(1, 1.2 * cm),
        P("Objetivo", "Heading2"),
        P(
            "Formular y comparar tres algoritmos supervisados de la familia de árboles "
            "utilizando un protocolo reproducible de clasificación binaria. El proyecto "
            "incluye los criterios de aprendizaje, el ajuste de hiperparámetros, la "
            "evaluación en datos no usados para entrenar y una aplicación interactiva.",
        ),
        P(
            "Conjunto experimental: Breast Cancer Wisconsin (Diagnostic), repositorio "
            "UCI Machine Learning Repository. Las métricas y conclusiones de este informe "
            "se generan desde la ejecución guardada en la carpeta <i>results</i>.",
        ),
        Spacer(1, 0.4 * cm),
        P("Fecha de generación: 27 de octubre de 2026", "Subtitle"),
        PageBreak(),
    ]

    story += [
        P("Resumen ejecutivo", "Heading1"),
        P(
            f"Se compararon CART, Random Forest y XGBoost con {metadata['n_samples']} "
            f"observaciones y {metadata['n_features']} variables. El mejor ROC-AUC en "
            f"el conjunto de prueba fue {fmt(best['roc_auc'])}, obtenido por "
            f"<b>{best['modelo']}</b>. Sus métricas para la clase positiva maligno fueron "
            f"precision {fmt(best['precision_maligno'])}, recall/sensibilidad "
            f"{fmt(best['recall_maligno'])} y F1 {fmt(best['f1_maligno'])}. "
            "Este resultado describe únicamente esta partición de prueba y no representa "
            "rendimiento clínico."
        ),
        P(
            "La comparación ilustra tres estrategias diferentes: un árbol CART es "
            "interpretable pero puede variar mucho con cambios en los datos; Random "
            "Forest promedia árboles aleatorizados para reducir varianza; XGBoost añade "
            "árboles secuencialmente y optimiza una pérdida con regularización. Se "
            "mantuvieron las mismas filas de prueba para los tres modelos.",
        ),
        P("1. Fuente y descripción del conjunto de datos", "Heading1"),
        P(
            "La fuente original es <b>Breast Cancer Wisconsin (Diagnostic), UCI Machine "
            "Learning Repository</b> [1]. La ficha de UCI describe 569 muestras, 30 "
            "características numéricas y una variable de diagnóstico; las mediciones se "
            "calcularon a partir de imágenes digitalizadas de aspiración con aguja fina "
            "(FNA) y describen núcleos celulares. La ficha informa que no hay valores "
            "faltantes. El conjunto fue donado a UCI el 31 de octubre de 1995."
        ),
        P(
            "Se cargó la copia que distribuye scikit-learn mediante "
            "<i>sklearn.datasets.load_breast_cancer</i>, que la documenta como una copia "
            "del conjunto UCI y expone una matriz de 569 × 30 [2]. El archivo "
            "<i>data/dataset.csv</i> conserva las 30 variables numéricas y la etiqueta "
            "recodificada; se excluye cualquier identificador, pues no es una variable "
            "predictiva. La etiqueta de scikit-learn usa 0=maligno y 1=benigno; se "
            "invirtió a 1=maligno (positivo) y 0=benigno para que recall y falsos "
            "negativos se interpreten de forma directa."
        ),
        P(
            f"En la copia analizada hay {metadata['class_counts']['benigno']} casos "
            f"benignos y {metadata['class_counts']['maligno']} malignos. El conjunto "
            "completo no se presenta como un conjunto clínico representativo de la "
            "población actual; es una referencia académica pequeña y tabular."
        ),
        P("2. Preparación de datos y protocolo experimental", "Heading1"),
        P(
            f"Se realizó una partición estratificada, con semilla fija "
            f"{metadata['seed']}: {metadata['n_train']} filas para entrenamiento "
            f"({100 * (1-metadata['test_size']):.0f}%) y {metadata['n_test']} para prueba "
            f"({100 * metadata['test_size']:.0f}%). Estratificar conserva aproximadamente "
            "la prevalencia de ambas clases. La prueba se mantuvo fuera de la selección "
            "de parámetros y se usó una sola vez para calcular el resultado final."
        ),
        P(
            f"Sobre el entrenamiento se aplicó RandomizedSearchCV con "
            f"{metadata['random_search_iterations']} combinaciones aleatorias por modelo "
            f"y validación cruzada estratificada de {metadata['cv_splits']} pliegues "
            f"(semilla {metadata['seed']}). El criterio de selección fue ROC-AUC "
            "promedio. Cada búsqueda volvió a ajustar el mejor estimador sobre todo el "
            "conjunto de entrenamiento. La clase positiva y el umbral final se fijaron "
            "antes de observar los resultados: maligno y 0.5, respectivamente."
        ),
        P(
            "No se normalizaron las variables: los modelos basados en árboles eligen "
            "umbrales de partición y no dependen de la escala euclidiana. Se verifican "
            "datos faltantes (UCI reporta ninguno). No se emplearon transformaciones "
            "aprendidas del conjunto completo, evitando fuga de información."
        ),
        P(
            "Configuración explorada: CART (criterio Gini/entropía/log-loss, profundidad, "
            "mínimo de muestras por división/hoja y poda ccp_alpha); Random Forest "
            "(200–700 árboles, máximo de variables, profundidad, mínimo de hoja/división, "
            "bootstrap y ponderación de clases); XGBoost (100–500 árboles, tasa de "
            "aprendizaje, profundidad, peso mínimo de hoja, subsampling de filas y "
            "variables, regularización L2 y gamma). La combinación aleatoria concreta "
            "y ganancia de validación se registran en "
            "<i>results/mejores_parametros.json</i>."
        ),
        P("3. Fundamentos matemáticos", "Heading1"),
        P("3.1 Árbol de decisión CART", "Heading2"),
        P(
            "CART (Classification and Regression Trees) construye un árbol binario. "
            "Cada nodo interno evalúa una regla del tipo xⱼ ≤ t; las hojas asignan una "
            "clase (o estiman una probabilidad). El aprendizaje es voraz: en cada nodo "
            "se elige la partición que más reduce la impureza ponderada, sin buscar "
            "exhaustivamente el árbol global óptimo [3, 4]."
        ),
        P("Gini(D) = 1 − Σₖ pₖ²     ·     Entropía(D) = −Σₖ pₖ log₂(pₖ)", "Formula"),
        P(
            "Para una división en subconjuntos Dₗ y Dᵣ, la ganancia es "
            "I(D) − (|Dₗ|/|D|)I(Dₗ) − (|Dᵣ|/|D|)I(Dᵣ). En clasificación binaria, "
            "pₖ es la proporción de ejemplos de la clase k en el nodo. Los parámetros "
            "de profundidad, mínimo de hoja y ccp_alpha limitan la complejidad y "
            "regularizan el sobreajuste. El algoritmo busca óptimos locales mediante "
            "búsquedas de umbrales; no garantiza el árbol global óptimo."
        ),
        P("3.2 Random Forest: bagging de árboles", "Heading2"),
        P(
            "Random Forest entrena T árboles con muestras bootstrap independientes: "
            "cada muestra se obtiene con reemplazo del entrenamiento. En cada nodo, "
            "el árbol considera además una selección aleatoria de variables. Para "
            "clasificación se combinan votos o probabilidades; aquí se usa el promedio "
            "de probabilidades estimadas por hoja [5]."
        ),
        P("p̂(y=1 | x) = (1/T) Σₜ₌₁ᵀ p̂ₜ(y=1 | x)", "Formula"),
        P(
            "El promedio de árboles correlacionados imperfectamente tiende a reducir "
            "la varianza de un árbol individual. La aleatorización de filas y variables "
            "busca disminuir su correlación; no elimina el sesgo y puede reducir la "
            "interpretabilidad global. No se realiza una secuencia correctiva de errores: "
            "a diferencia de boosting, los árboles pueden entrenarse en paralelo."
        ),
        P("3.3 Gradient Boosting Decision Trees con XGBoost", "Heading2"),
        P(
            "XGBoost es un método de boosting aditivo. En cada iteración agrega un "
            "árbol que mejora el ajuste de las predicciones anteriores; los árboles "
            "se entrenan secuencialmente. Para clasificación binaria se suma una "
            "función de puntuación y se transforma con la función sigmoide [6]."
        ),
        P("Fₘ(x) = Fₘ₋₁(x) + η fₘ(x) = Σₜ₌₁ᵐ η fₜ(x)     ·     p̂(y=1|x)=1/(1+exp(−Fₘ(x)))", "Formula"),
        P(
            "La función objetivo combina la pérdida logística con penalización de "
            "complejidad: Obj = Σᵢ ℓ(yᵢ, F(xᵢ)) + Σₜ Ω(fₜ), donde "
            "Ω(f) = γT + (λ/2)Σⱼwⱼ². T es el número de hojas y wⱼ sus puntuaciones. "
            "Para y∈{0,1}, ℓ = −[y log(p) + (1−y)log(1−p)]. XGBoost aproxima el "
            "objetivo por una expansión de Taylor de segundo orden y usa gradientes "
            "y hessianos para puntuar divisiones; γ, λ, profundidad, tasa η y "
            "subsampling controlan capacidad y regularización. La tasa η escala la "
            "contribución del árbol añadido."
        ),
        P(
            "Las implementaciones representan estrategias diferentes: CART usa "
            "particiones voraces; Random Forest agrega modelos con bagging y "
            "aleatorización; XGBoost minimiza iterativamente el error residual en el "
            "espacio de funciones. El resultado depende de los datos, el criterio "
            "y los hiperparámetros, no hay un ganador universal."
        ),
    ]

    story += [
        PageBreak(),
        P("4. Métricas de evaluación", "Heading1"),
        P(
            "La matriz de confusión se cuenta con maligno como clase positiva: TP son "
            "malignos correctamente detectados, FN malignos clasificados como benignos, "
            "FP benignos clasificados como malignos y TN benignos correctamente "
            "clasificados. Un FN puede ser especialmente costoso en un contexto de "
            "cribado, por lo que recall se interpreta junto con precisión y falsos "
            "positivos; no se optimizó ni modificó un umbral clínico."
        ),
        P(
            "Recall = TP/(TP+FN) · Precision = TP/(TP+FP) · "
            "F1 = 2·Precision·Recall/(Precision+Recall)", "Formula"
        ),
        P(
            "ROC-AUC resume la capacidad de ordenar positivos por encima de negativos "
            "en distintos umbrales; no es exactitud ni sustituye la sensibilidad en un "
            "umbral operativo. Exactitud balanceada = (sensibilidad + especificidad)/2. "
            "También se muestra exactitud global, pero esta métrica sola puede ocultar "
            "errores en la clase minoritaria."
        ),
        P("5. Resultados medidos", "Heading1"),
        P(
            "La siguiente tabla corresponde exclusivamente al conjunto de prueba "
            "reservado. ROC-AUC de validación cruzada se informa aparte y sirvió para "
            "seleccionar hiperparámetros."
        ),
    ]

    headers = [
        "Modelo", "ROC-AUC<br/>prueba", "Precision<br/>maligno", "Recall<br/>maligno",
        "F1<br/>maligno", "Exactitud", "Bal.<br/>exactitud",
    ]
    rows = [[P(item, "TableHeader") for item in headers]]
    for _, row in metrics.iterrows():
        rows.append(
            [
                P(str(row["modelo"]), "Table"),
                P(fmt(row["roc_auc"]), "Table"),
                P(fmt(row["precision_maligno"]), "Table"),
                P(fmt(row["recall_maligno"]), "Table"),
                P(fmt(row["f1_maligno"]), "Table"),
                P(fmt(row["exactitud"]), "Table"),
                P(fmt(row["exactitud_balanceada"]), "Table"),
            ]
        )
    story.append(make_table(rows, [3.0 * cm, 2.1 * cm, 2.4 * cm, 2.4 * cm, 1.7 * cm, 1.7 * cm, 2.0 * cm]))
    story.append(Spacer(1, 0.25 * cm))

    for _, row in metrics.iterrows():
        story.append(
            P(
                f"<b>{row['modelo']} — matriz de confusión</b> (filas: clase real; "
                f"columnas: predicción; orden benigno, maligno): "
                f"[[TN={int(row['verdaderos_negativos'])}, FP={int(row['falsos_positivos'])}], "
                f"[FN={int(row['falsos_negativos'])}, TP={int(row['verdaderos_positivos'])}]]. "
                f"ROC-AUC CV={fmt(row['mejor_roc_auc_cv'])}; tiempo de ajuste "
                f"{row['tiempo_ajuste_segundos']:.2f} s (dependiente del hardware)."
            )
        )

    story += [
        P(
            "Los parámetros seleccionados por CV y la ROC-AUC promedio aparecen debajo. "
            "La métrica de CV no debe confundirse con la puntuación final de prueba."
        ),
    ]
    for model_name, values in params.items():
        story.append(
            P(
                f"<b>{model_name}</b>: ROC-AUC CV = "
                f"{values['best_cv_roc_auc']:.4f}; parámetros = "
                f"<font name=\"{fonts}\">{json.dumps(values['best_params'], ensure_ascii=False)}</font>",
                "Small",
            )
        )

    story += [
        PageBreak(),
        P("6. Visualizaciones", "Heading1"),
        P(
            "Las curvas ROC y matrices corresponden a predicciones del conjunto de "
            "prueba. Las importancias son propias de cada implementación (reducción "
            "de impureza para árboles de scikit-learn y la importancia expuesta por "
            "XGBoost); sus escalas no se deben comparar directamente ni interpretarse "
            "como efectos causales."
        ),
    ]
    add_image(story, "curvas_roc.png", 17 * cm, 11 * cm)
    story.append(P("Figura 1. Curvas ROC de los tres modelos en los mismos datos de prueba.", "Small"))
    add_image(story, "matrices_confusion.png", 17 * cm, 8 * cm)
    story.append(P("Figura 2. Matrices de confusión; la clase positiva es maligno.", "Small"))
    add_image(story, "comparacion_metricas.png", 17 * cm, 9.5 * cm)
    story.append(P("Figura 3. Comparación de ROC-AUC, F1, recall y precisión en prueba.", "Small"))
    story.append(PageBreak())
    story += [
        P("6.1 Importancia de variables y árbol interpretable", "Heading2"),
        P(
            "Las importancias muestran qué características fueron útiles para las "
            "particiones de estos modelos entrenados. CART se visualiza hasta "
            "profundidad 3 únicamente para facilitar la lectura; la figura no recorta "
            "el estimador utilizado para las métricas. Para una explicación de un caso "
            "individual se debe recorrer sus reglas hasta una hoja y tener en cuenta "
            "que el árbol puede ser inestable ante cambios en los datos."
        ),
    ]
    add_image(story, "importancia_variables.png", 17 * cm, 8 * cm)
    story.append(P("Figura 4. Diez variables de mayor importancia según cada modelo.", "Small"))
    add_image(story, "arbol_cart.png", 17 * cm, 10 * cm)
    story.append(P("Figura 5. Árbol CART seleccionado, truncado visualmente a profundidad 3.", "Small"))

    story += [
        PageBreak(),
        P("7. Aplicaciones y análisis crítico", "Heading1"),
        P("CART — decisiones auditables", "Heading2"),
        P(
            "Un árbol pequeño puede apoyar la priorización de casos para revisión "
            "humana o la clasificación preliminar de registros estructurados, porque "
            "cada recorrido puede describirse con reglas. La profundidad limitada "
            "facilita comunicarlo, pero puede sacrificar desempeño y estabilidad. "
            "En salud, una regla aprendida de esta base histórica no es una guía "
            "clínica; debe validarse en población y entorno de uso."
        ),
        P("Random Forest — patrones tabulares y robustez", "Heading2"),
        P(
            "El bagging suele ser útil en tareas tabulares como priorización de alertas "
            "de fraude, control de calidad o estimación de riesgo, cuando interesa "
            "estabilizar árboles individuales y modelar interacciones no lineales. "
            "El promedio hace menos transparente la explicación de una predicción "
            "individual y exige considerar falsos positivos, deriva y coste operativo."
        ),
        P("XGBoost — ranking de riesgo tabular", "Heading2"),
        P(
            "El boosting regularizado es un candidato frecuente para clasificación "
            "tabular de alto rendimiento, por ejemplo para ordenar transacciones para "
            "revisión antifraude o priorizar expedientes. Puede capturar interacciones "
            "complejas, pero su ajuste es más delicado y su salida no es una explicación "
            "causal. Importancias o explicaciones locales requieren análisis adicional "
            "y validación."
        ),
        P(
            "En el experimento real de este trabajo, los tres algoritmos se evaluaron "
            "únicamente sobre el mismo problema WDBC. Los casos de fraude y riesgo "
            "mencionados son aplicaciones ilustrativas, no ensayos adicionales ni "
            "resultados medidos en esos dominios."
        ),
        P("8. Simulación interactiva y entregables", "Heading1"),
        P(
            "La interfaz <i>app.py</i> permite seleccionar CART, Random Forest o "
            "XGBoost y obtener predicciones para ejemplos reservados de prueba o filas "
            "de un CSV con las 30 variables numéricas. Verifica columnas requeridas, "
            "valores numéricos y faltantes; presenta la clase y la probabilidad "
            "estimada de malignidad. Esa probabilidad no se interpreta como riesgo "
            "clínico calibrado."
        ),
        P(
            "El código reproducible se organiza en <i>src/experiment.py</i> y "
            "<i>src/generate_report.py</i>; los artefactos generados quedan en "
            "<i>data/</i> y <i>results/</i>. Para ejecutar: instalar "
            "<i>requirements.txt</i>, correr <i>python src\\experiment.py</i>, luego "
            "<i>python src\\generate_report.py</i> y abrir <i>streamlit run app.py</i>."
        ),
        P("9. Limitaciones, uso responsable y próximos pasos", "Heading1"),
        P(
            "El dataset es pequeño, histórico y procede de una fuente de imágenes y "
            "población específicas; no contiene suficiente contexto clínico para "
            "decisiones reales. Una partición aleatoria no sustituye validación externa "
            "ni prueba temporal. ROC-AUC no define el umbral adecuado; los costes de "
            "falsos negativos y falsos positivos se deben acordar con profesionales "
            "antes de cualquier uso. La probabilidad de un árbol/ensamble puede no "
            "estar calibrada. No se midieron intervalos de confianza, calibración, "
            "impacto de subgrupos ni robustez ante deriva. No se debe usar esta "
            "simulación en diagnóstico, tratamiento o triaje clínico."
        ),
        P(
            "Siguientes pasos para un estudio no académico: validación externa "
            "independiente; estimación de incertidumbre; análisis por subgrupos; "
            "calibración probabilística; selección prospectiva del umbral mediante "
            "análisis de costes y revisión ética/regulatoria."
        ),
        P("10. Conclusiones", "Heading1"),
        P(
            f"En esta partición, {best['modelo']} obtuvo el mayor ROC-AUC de prueba "
            f"({fmt(best['roc_auc'])}). El resultado no implica que siempre supere a "
            "los otros modelos ni establece superioridad clínica. El modelo con mayor "
            "recall de malignos fue "
            f"{metrics.loc[metrics['recall_maligno'].idxmax(), 'modelo']} "
            f"({fmt(metrics['recall_maligno'].max())}); por eso la elección depende "
            "del balance entre sensibilidad, precisión, explicabilidad, costes y "
            "validación apropiada. La evaluación común y la prueba separada permiten "
            "una comparación más justa que entrenar cada modelo con particiones "
            "diferentes."
        ),
        P("Referencias", "Heading1"),
    ]

    references = [
        "[1] UCI Machine Learning Repository. Breast Cancer Wisconsin (Diagnostic). "
        "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "[2] scikit-learn. load_breast_cancer: dataset loader and source attribution. "
        "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
        "[3] scikit-learn. Decision Trees — classification, impurity and pruning. "
        "https://scikit-learn.org/stable/modules/tree.html",
        "[4] Breiman, L., Friedman, J. H., Olshen, R. A. & Stone, C. J. (1984). "
        "Classification and Regression Trees. Chapman & Hall/CRC.",
        "[5] Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32. "
        "https://doi.org/10.1023/A:1010933404324",
        "[6] XGBoost contributors. Introduction to Boosted Trees. "
        "https://xgboost.readthedocs.io/en/stable/tutorials/model.html",
        "[7] Friedman, J. H. (2001). Greedy Function Approximation: A Gradient "
        "Boosting Machine. Annals of Statistics, 29(5), 1189–1232. "
        "https://doi.org/10.1214/aos/1013203451",
        "[8] scikit-learn. Model evaluation: quantifying the quality of predictions. "
        "https://scikit-learn.org/stable/modules/model_evaluation.html",
    ]
    for reference in references:
        story.append(P(reference, "Small"))
    story += [
        Spacer(1, 0.25 * cm),
        P(
            f"Entorno registrado en la ejecución: Python {metadata['versions']['python']}, "
            f"scikit-learn {metadata['versions']['scikit_learn']}, "
            f"XGBoost {metadata['versions']['xgboost']}, pandas "
            f"{metadata['versions']['pandas']}, NumPy {metadata['versions']['numpy']}. "
            "Las ejecuciones pueden variar levemente según versiones de bibliotecas "
            "y hardware.",
            "Small",
        ),
    ]

    def footer(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(fonts, 8)
        canvas.setFillColor(MID_GRAY)
        canvas.drawString(1.65 * cm, 0.95 * cm, "Equipo 1 · Informe académico")
        canvas.drawRightString(A4[0] - 1.65 * cm, 0.95 * cm, f"Página {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Informe generado: {OUTPUT}")


if __name__ == "__main__":
    register_fonts()
    build_report()
