# CHANGELOG

Todos los cambios relevantes realizados durante el desarrollo del proyecto fueron documentados en este archivo siguiendo principios de trazabilidad y reproducibilidad.

---

# [1.0.0] - Inicialización del Proyecto

## Agregado

* Creación del repositorio Git.
* Configuración inicial del proyecto en Visual Studio Code.
* Definición de estructura de carpetas:

  * datos/
  * fuentes/
  * modelos/
  * reports/
  * mlruns/
* Creación de archivos base:

  * README.md
  * CHANGELOG.md
  * requirements.txt

---

# [1.1.0] - Integración del Dataset

## Agregado

* Descarga del dataset Hotel Booking Demand desde Kaggle.

* Incorporación del archivo:

  ```text
  hotel_bookings.csv
  ```

* Documentación de la fuente de datos.

* Definición de la variable objetivo:

  ```text
  is_canceled
  ```

## Fuente

https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

---

# [1.2.0] - Limpieza y Preparación de Datos

## Agregado

* Desarrollo del script:

  ```text
  fuentes/datos_prep.py
  ```

* Eliminación de registros duplicados.

* Tratamiento de valores nulos.

* Conversión de tipos de datos.

* Aplicación de reglas de negocio.

* Validación de consistencia de huéspedes.

## Modificado

* Creación de variable derivada:

  ```text
  total_nights
  ```

* Generación del dataset limpio:

  ```text
  hotel_bookings_clean.csv
  ```

---

# [1.3.0] - Ingeniería de Características

## Agregado

* Identificación de variables numéricas.
* Identificación de variables categóricas.
* Implementación de escalado mediante StandardScaler.
* Implementación de codificación One-Hot Encoding.

## Mejoras

* Automatización del pipeline de transformación.
* Compatibilidad con nuevas categorías no observadas durante entrenamiento.

---

# [1.4.0] - Entrenamiento del Modelo Random Forest

## Agregado

* Desarrollo del pipeline de entrenamiento.
* División Train/Test.
* Configuración de validación cruzada estratificada.
* Implementación de GridSearchCV.

## Parámetros evaluados

* n_estimators
* max_depth
* min_samples_split

## Métricas registradas

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# [1.5.0] - Entrenamiento del Modelo XGBoost

## Agregado

* Integración de XGBoost.
* Ajuste de hiperparámetros mediante Grid Search.
* Evaluación bajo las mismas condiciones experimentales que Random Forest.

## Parámetros evaluados

* n_estimators
* max_depth
* learning_rate

## Objetivo

Comparar desempeño predictivo y eficiencia computacional frente a Random Forest.

---

# [1.6.0] - Integración de MLflow

## Agregado

* Configuración del experimento:

  ```text
  Actividad5_Hotel_Booking_Demand
  ```

* Registro automático de:

  * Parámetros.
  * Métricas.
  * Modelos.
  * Artefactos.

## Artefactos generados

* Matrices de confusión.
* Curvas ROC.
* Modelos serializados.
* Resultados comparativos.

## Beneficios

* Reproducibilidad.
* Trazabilidad de experimentos.
* Comparación estructurada de ejecuciones.

---

# [1.7.0] - Evaluación Comparativa de Modelos

## Resultados obtenidos

### Random Forest

| Métrica   | Valor  |
| --------- | ------ |
| Accuracy  | 0.8518 |
| Precision | 0.7859 |
| Recall    | 0.6343 |
| F1 Score  | 0.7020 |
| ROC-AUC   | 0.9140 |

### XGBoost

| Métrica   | Valor  |
| --------- | ------ |
| Accuracy  | 0.8490 |
| Precision | 0.7599 |
| Recall    | 0.6597 |
| F1 Score  | 0.7063 |
| ROC-AUC   | 0.9173 |

## Decisión

Se seleccionó XGBoost como modelo final debido a:

* Mejor Recall.
* Mejor F1 Score.
* Mejor ROC-AUC.
* Menor tiempo de entrenamiento.

---

# [1.8.0] - Documentación y Reproducibilidad

## Agregado

* Elaboración de README.md.
* Documentación del flujo completo de ejecución.
* Instrucciones de instalación.
* Instrucciones para reproducir experimentos.
* Registro del entorno de dependencias.

## Estado Final

Proyecto reproducible y documentado bajo prácticas básicas de MLOps utilizando:

* Git
* GitHub
* MLflow
* Scikit-Learn
* XGBoost
* Visual Studio Code
