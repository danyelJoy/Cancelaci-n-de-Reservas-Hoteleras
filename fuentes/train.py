import mlflow
import mlflow.sklearn
import mlflow.xgboost

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


RUTA_DATOS = Path("datos/datos_limp/hotel_bookings_clean.csv")
RUTA_FIGURAS = Path("reports/figuras")
RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)


def cargar_datos():
    return pd.read_csv(RUTA_DATOS)


def preparar_variables(df):
    y = df["is_canceled"]

    columnas_eliminar = [
        "is_canceled",
        "reservation_status",
        "reservation_status_date",
    ]

    X = df.drop(columns=columnas_eliminar)

    columnas_numericas = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    columnas_categoricas = X.select_dtypes(include=["object"]).columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), columnas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
        ]
    )

    return X, y, preprocesador


def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metricas


def guardar_matriz_confusion(modelo, X_test, y_test, nombre_modelo):
    y_pred = modelo.predict(X_test)

    matriz = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matriz)

    display.plot()
    plt.title(f"Matriz de confusión - {nombre_modelo}")

    ruta = RUTA_FIGURAS / f"matriz_confusion_{nombre_modelo}.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()

    return ruta


def guardar_curva_roc(modelo, X_test, y_test, nombre_modelo):
    display = RocCurveDisplay.from_estimator(modelo, X_test, y_test)
    plt.title(f"Curva ROC - {nombre_modelo}")

    ruta = RUTA_FIGURAS / f"curva_roc_{nombre_modelo}.png"
    plt.savefig(ruta, bbox_inches="tight")
    plt.close()

    return ruta


def entrenar_y_registrar(nombre_modelo, modelo, parametros, X_train, X_test, y_train, y_test, preprocesador):
    pipeline = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", modelo),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run(run_name=nombre_modelo):
        grid_search.fit(X_train, y_train)

        mejor_modelo = grid_search.best_estimator_
        metricas = evaluar_modelo(mejor_modelo, X_test, y_test)

        mlflow.log_param("modelo", nombre_modelo)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_params(grid_search.best_params_)

        for nombre_metrica, valor in metricas.items():
            mlflow.log_metric(nombre_metrica, valor)

        ruta_matriz = guardar_matriz_confusion(mejor_modelo, X_test, y_test, nombre_modelo)
        ruta_roc = guardar_curva_roc(mejor_modelo, X_test, y_test, nombre_modelo)

        mlflow.log_artifact(str(ruta_matriz))
        mlflow.log_artifact(str(ruta_roc))

        if nombre_modelo == "xgboost":
            mlflow.xgboost.log_model(mejor_modelo.named_steps["modelo"], "modelo_xgboost")
        else:
            mlflow.sklearn.log_model(mejor_modelo, "modelo_sklearn")

        print(f"\nModelo: {nombre_modelo}")
        print(f"Mejores parámetros: {grid_search.best_params_}")
        print("Métricas:")
        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

        return metricas


def main():
    mlflow.set_experiment("Actividad5_Hotel_Booking_Demand")

    df = cargar_datos()
    X, y, preprocesador = preparar_variables(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    modelos = {
        "random_forest": {
            "modelo": RandomForestClassifier(random_state=42, n_jobs=-1),
            "parametros": {
                "modelo__n_estimators": [100],
                "modelo__max_depth": [10, 20, None],
                "modelo__min_samples_split": [2, 5],
            },
        },
        "xgboost": {
            "modelo": XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "parametros": {
                "modelo__n_estimators": [100],
                "modelo__max_depth": [3, 5, 7],
                "modelo__learning_rate": [0.05, 0.1],
            },
        },
    }

    resultados = {}

    for nombre_modelo, config in modelos.items():
        metricas = entrenar_y_registrar(
            nombre_modelo=nombre_modelo,
            modelo=config["modelo"],
            parametros=config["parametros"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocesador=preprocesador,
        )

        resultados[nombre_modelo] = metricas

    resultados_df = pd.DataFrame(resultados).T
    resultados_df.to_csv("reports/resultados_modelos.csv")

    print("\nComparación final:")
    print(resultados_df)


if __name__ == "__main__":
    main()