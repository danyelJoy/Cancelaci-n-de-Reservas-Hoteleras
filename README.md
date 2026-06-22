# Actividad 5 – Entrenamiento, Ajuste y Registro de Modelos con MLflow

## Descripción del Proyecto

Este proyecto implementa un flujo completo de Machine Learning orientado a la predicción de cancelaciones de reservas hoteleras utilizando el dataset **Hotel Booking Demand** de Kaggle.

El objetivo principal consiste en comparar el desempeño de dos algoritmos de clasificación supervisada:

* Random Forest
* XGBoost

Para garantizar la reproducibilidad y trazabilidad del experimento se empleó:

* Git y GitHub para control de versiones.
* MLflow para seguimiento de experimentos.
* GridSearchCV para ajuste de hiperparámetros.
* Validación cruzada de 5 particiones (5-Fold Cross Validation).

---

## Objetivo

Desarrollar un pipeline reproducible de Machine Learning que permita:

1. Preparar y limpiar datos.
2. Entrenar múltiples modelos.
3. Ajustar hiperparámetros mediante Grid Search.
4. Evaluar modelos utilizando métricas estandarizadas.
5. Registrar experimentos mediante MLflow.
6. Comparar resultados para seleccionar el mejor modelo.

---

## Dataset Utilizado

### Hotel Booking Demand Dataset

Fuente:

https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

Descripción:

El conjunto de datos contiene información histórica de reservas realizadas en hoteles urbanos y resorts.

Variable objetivo:

```text
is_canceled
```

Valores:

```text
0 = Reserva no cancelada
1 = Reserva cancelada
```

---

## Estructura del Proyecto

```text
Actividad5/

│── datos/
│   ├── datos_ini/
│   │   └── hotel_bookings.csv
│   │
│   └── datos_limp/
│       └── hotel_bookings_clean.csv
│
│── fuentes/
│   ├── datos_prep.py
│   ├── train.py
│   └── entrena.ipynb
│
│── modelos/
│
│── reports/
│   ├── resultados_modelos.csv
│   └── figuras/
│
│── mlruns/
│
│── README.md
│── CHANGELOG.md
│── requirements.txt
```

---

## Preparación de Datos

Las actividades realizadas durante el preprocesamiento incluyen:

* Eliminación de registros duplicados.
* Tratamiento de valores nulos.
* Conversión de tipos de datos.
* Aplicación de reglas de negocio.
* Generación de nuevas variables derivadas.
* Escalado de variables numéricas.
* Codificación One-Hot para variables categóricas.

Asimismo, se creó una variable denominada:

```text
total_nights
```

correspondiente a la suma de noches entre semana y fines de semana.

---

## Modelos Implementados

### Random Forest

Algoritmo basado en ensamblado de árboles de decisión mediante bagging.

Parámetros ajustados:

* n_estimators
* max_depth
* min_samples_split

---

### XGBoost

Algoritmo de Gradient Boosting optimizado para problemas supervisados.

Parámetros ajustados:

* n_estimators
* max_depth
* learning_rate

---

## Estrategia de Validación

Se utilizó:

```text
Stratified K-Fold Cross Validation
```

Configuración:

```text
Número de folds: 5
Shuffle: True
Random State: 42
```

Adicionalmente se implementó:

```text
GridSearchCV
```

para búsqueda sistemática de hiperparámetros.

---

## Métricas de Evaluación

Los modelos fueron evaluados mediante:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

Estas métricas permiten evaluar tanto la capacidad predictiva global como la habilidad para identificar correctamente reservas canceladas.

---

## Resultados Obtenidos

| Modelo        | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------- | -------- | --------- | ------ | -------- | ------- |
| Random Forest | 0.8518   | 0.7859    | 0.6343 | 0.7020   | 0.9140  |
| XGBoost       | 0.8490   | 0.7599    | 0.6597 | 0.7063   | 0.9173  |

---

## Modelo Seleccionado

Aunque Random Forest obtuvo una Accuracy ligeramente superior, XGBoost presentó mejores resultados en:

* Recall
* F1 Score
* ROC-AUC

Por esta razón se seleccionó XGBoost como modelo final.

Además, el tiempo de entrenamiento fue significativamente menor:

```text
Random Forest: 21.5 minutos
XGBoost: 47 segundos
```

Lo anterior representa una ventaja importante desde una perspectiva de escalabilidad y MLOps.

---

## Seguimiento de Experimentos con MLflow

Todos los experimentos fueron registrados mediante MLflow.

Se almacenó:

* Parámetros de entrenamiento.
* Hiperparámetros optimizados.
* Métricas de evaluación.
* Matrices de confusión.
* Curvas ROC.
* Modelos entrenados.

Para iniciar la interfaz de MLflow:

```bash
mlflow ui
```

Acceder desde:

```text
http://127.0.0.1:5000
```

---

## Instalación

Crear entorno virtual:

```bash
python -m venv venv
```

Activar entorno:

```bash
source venv/Scripts/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución

### Preparación de datos

```bash
python fuentes/datos_prep.py
```

### Entrenamiento de modelos

```bash
python fuentes/train.py
```

### Visualización de experimentos

```bash
mlflow ui
```

---

## Tecnologías Utilizadas

* Python 3.x
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* MLflow
* Git
* GitHub
* Visual Studio Code

---

## Autor
D.V.R.M

Actividad desarrollada para la materia de Gestión de Proyectos de Inteligencia Artificial.

Master en Inteligencia Artificial.
