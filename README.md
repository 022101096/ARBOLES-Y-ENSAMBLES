# Equipo 1 — Árboles y ensambles

Proyecto académico de clasificación binaria que compara CART, Random Forest y
Gradient Boosting con XGBoost sobre Breast Cancer Wisconsin (Diagnostic).

> **Uso académico:** los datos y la aplicación son demostrativos. No son un
> dispositivo médico ni deben utilizarse para diagnosticar o tratar pacientes.

## Reproducir

Desde esta carpeta, con Python 3.10 o posterior:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\experiment.py
python src\generate_report.py
streamlit run app.py
```

`experiment.py` carga la copia de scikit-learn del conjunto de datos de UCI,
crea una partición estratificada 80/20, optimiza cada modelo mediante
`RandomizedSearchCV` y validación cruzada estratificada de cinco pliegues, y
guarda los modelos y resultados. `generate_report.py` produce
`informe_tecnico.pdf` con el método, resultados observados, figuras y fuentes.
La aplicación requiere haber ejecutado primero el experimento.

## Estructura

- `src/experiment.py`: carga, partición, búsqueda, evaluación, modelos y figuras.
- `src/generate_report.py`: informe final en español y exportación a PDF.
- `app.py`: simulador interactivo Streamlit.
- `data/`: copia local del conjunto y ejemplos de prueba generados.
- `results/`: modelos entrenados, métricas, parámetros y gráficas.
- `informe_tecnico.pdf`: informe generado después de entrenar.

## Convenciones de evaluación

La etiqueta positiva es **maligno** (1); benigno es 0. La búsqueda selecciona
hiperparámetros maximizando ROC-AUC en los pliegues de entrenamiento. El umbral
de clasificación queda en 0.5. El conjunto de prueba se reserva para la
evaluación final y no participa en el ajuste. Los árboles no requieren
escalamiento y el conjunto no contiene valores faltantes.

El conjunto tiene 569 observaciones y 30 mediciones numéricas derivadas de
imágenes de aspiración con aguja fina. Su fuente original es UCI Machine
Learning Repository; scikit-learn distribuye una copia de ese conjunto.
Consultar las referencias completas en el informe.
