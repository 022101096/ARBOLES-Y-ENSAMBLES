"""Entrena y compara CART, Random Forest y XGBoost sobre WDBC."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SEED = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
N_ITER = 20
POSITIVE_LABEL = 1  # malignant
CLASS_NAMES = ["Benigno", "Maligno"]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve variables numéricas y etiqueta: 1=maligno, 0=benigno."""
    bundle = load_breast_cancer(as_frame=True)
    X = bundle.data.copy()
    # En la copia de scikit-learn, el índice 0 es maligno y el 1 benigno.
    y = pd.Series(
        (bundle.target == 0).astype("int8"),
        index=X.index,
        name="maligno",
    )
    return X, y


def model_spaces() -> dict[str, tuple[object, dict[str, list]]]:
    return {
        "CART": (
            DecisionTreeClassifier(random_state=SEED),
            {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None, 2, 3, 4, 5, 6, 8],
                "min_samples_split": [2, 5, 10, 15],
                "min_samples_leaf": [1, 2, 4, 6, 10],
                "ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {
                "n_estimators": [200, 300, 500, 700],
                "max_features": ["sqrt", 0.5, None],
                "max_depth": [None, 4, 6, 8, 12],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
                "class_weight": [None, "balanced_subsample"],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=1,
                verbosity=0,
            ),
            {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4, 5],
                "min_child_weight": [1, 2, 4, 6],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [1, 2, 5, 10],
                "gamma": [0, 0.1, 0.5],
            },
        ),
    }


def save_figures(
    fitted: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc = roc_auc_score(y_test, probabilities)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar")
    ax.set(title="Curvas ROC en el conjunto de prueba", xlabel="Tasa de falsos positivos",
           ylabel="Sensibilidad (maligno)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "curvas_roc.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(fitted), figsize=(15, 4.8))
    for ax, (name, model) in zip(axes, fitted.items()):
        prediction = model.predict(X_test)
        matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        )
        ax.set(title=name, xlabel="Predicción", ylabel="Valor real")
    fig.suptitle("Matrices de confusión (prueba)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "matrices_confusion.png", dpi=180)
    plt.close(fig)

    metric_names = ["ROC-AUC", "F1 maligno", "Recall maligno", "Precision maligno"]
    cols = ["roc_auc", "f1_maligno", "recall_maligno", "precision_maligno"]
    chart = metrics.set_index("modelo")[cols].rename(columns=dict(zip(cols, metric_names)))
    ax = chart.plot(kind="bar", figsize=(10, 5), ylim=(0, 1.05), rot=0)
    ax.set(title="Comparación de métricas en prueba", ylabel="Puntuación")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparacion_metricas.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, len(fitted), figsize=(16, 6))
    for ax, (name, model) in zip(axes, fitted.items()):
        values = np.asarray(model.feature_importances_)
        top = np.argsort(values)[-10:]
        ax.barh(X_test.columns[top], values[top], color="#3977a8")
        ax.set(title=name, xlabel="Importancia del modelo")
    fig.suptitle("Diez variables con mayor importancia por modelo")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "importancia_variables.png", dpi=180)
    plt.close(fig)

    tree = fitted["CART"]
    fig, ax = plt.subplots(figsize=(22, 12))
    plot_tree(
        tree,
        feature_names=list(X_test.columns),
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("CART seleccionado (visualización limitada a profundidad 3)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "arbol_cart.png", dpi=180)
    plt.close(fig)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y = load_dataset()
    dataset = X.copy()
    dataset["maligno"] = y
    dataset.to_csv(DATA_DIR / "dataset.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    X_test.to_csv(DATA_DIR / "ejemplos_prueba.csv", index=False)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    fitted: dict[str, object] = {}
    metric_rows: list[dict] = []
    search_summary: dict[str, dict] = {}
    print(
        f"Datos: {len(X)} filas, {X.shape[1]} variables; "
        f"entrenamiento={len(X_train)}, prueba={len(X_test)}."
    )
    print(f"Versiones: scikit-learn {sklearn.__version__}, XGBoost {xgboost.__version__}")

    for name, (estimator, params) in model_spaces().items():
        print(f"\nAjustando {name}...")
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=params,
            n_iter=N_ITER,
            scoring="roc_auc",
            cv=cv,
            refit=True,
            random_state=SEED,
            # Serial search avoids native XGBoost/OpenMP crashes in Windows worker processes.
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        model = search.best_estimator_
        fitted[name] = model

        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, POSITIVE_LABEL]
        matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        row = {
            "modelo": name,
            "roc_auc": roc_auc_score(y_test, probability),
            "precision_maligno": precision_score(y_test, prediction, zero_division=0),
            "recall_maligno": recall_score(y_test, prediction, zero_division=0),
            "f1_maligno": f1_score(y_test, prediction, zero_division=0),
            "exactitud": accuracy_score(y_test, prediction),
            "exactitud_balanceada": balanced_accuracy_score(y_test, prediction),
            "verdaderos_negativos": int(tn),
            "falsos_positivos": int(fp),
            "falsos_negativos": int(fn),
            "verdaderos_positivos": int(tp),
            "mejor_roc_auc_cv": search.best_score_,
            "tiempo_ajuste_segundos": fit_seconds,
        }
        metric_rows.append(row)
        search_summary[name] = {
            "best_params": search.best_params_,
            "best_cv_roc_auc": float(search.best_score_),
        }
        dump(model, RESULTS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        print(
            f"  ROC-AUC={row['roc_auc']:.4f}, "
            f"recall maligno={row['recall_maligno']:.4f}, "
            f"F1 maligno={row['f1_maligno']:.4f}"
        )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_DIR / "metricas.csv", index=False, float_format="%.6f")
    with (RESULTS_DIR / "mejores_parametros.json").open("w", encoding="utf-8") as file:
        json.dump(search_summary, file, ensure_ascii=False, indent=2)
    metadata = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "random_search_iterations": N_ITER,
        "search_scoring": "roc_auc",
        "positive_class": "maligno",
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_counts": {
            "benigno": int((y == 0).sum()),
            "maligno": int((y == 1).sum()),
        },
        "versions": {
            "python": __import__("platform").python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with (RESULTS_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    save_figures(fitted, X_test, y_test, metrics)
    print(f"\nResultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
