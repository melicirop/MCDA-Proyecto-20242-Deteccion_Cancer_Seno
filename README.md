# Detección de Cáncer en Imágenes Radiológicas con Modelos Combinados
Este proyecto implementa un modelo combinado de aprendizaje profundo y aprendizaje tabular para la detección de cáncer en imágenes radiológicas. Integra datos de imágenes y características clínicas para proporcionar predicciones más precisas.

## Descripción del Proyecto

La herramienta utiliza:
Redes neuronales convolucionales (CNN) para extraer características de imágenes.
Redes densas para procesar variables tabulares (edad, densidad mamaria, etc.).
Un modelo combinado que une ambas ramas para realizar una clasificación binaria: Cáncer o No Cáncer.
Además, se incluye una interfaz amigable desarrollada con Gradio para facilitar su uso por profesionales médicos.

El dataset para el entrenamiento fue tomado de la competencia [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)

## Estructura del proyecto

* Notebooks:
    * En la carpeta Notebooks podrá encontrar los notebooks documentados para fines educativos, deben estudiarse y ejecutarse en el orden en el que se encuentran enumerados

## Caracteristicas

* Procesamiento de Imágenes:

    * Conversión de imágenes en formato DICOM a PNG.
    * Generación de mapas de calor con el modelo VGG16 preentrenado.
    * Redimensionamiento y normalización de imágenes.

* Procesamiento de Datos Tabulares:

    * Escalado de características clínicas (edad, densidad, etc.) mediante StandardScaler.
    * Combinación con características extraídas de imágenes.
    * Manejo de nulos en caracteristica relevante: Simulación de Monte Carlos básica.

* Entrenamiento y Evaluación:

    * Modelo combinado con ResNet18 como rama de imágenes.
    * Métrica de evaluación: F1-Score.
    * Generación de matriz de confusión para evaluar el rendimiento.

* Despliegue:

    * Interfaz amigable en Gradio para subir imágenes y características clínicas.
    * Predicción de clase (Cáncer o No Cáncer) con confianza asociada.

## Tecnologías Utilizadas

Python version 3.11.7 (Librerías principales):
* PyTorch
* torchvision
* TensorFlow/Keras (para generación de heatmaps)
* Scikit-learn
* Gradio
* OpenCV
* Pillow

## Requisitos

1. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

2. Estructura esperada del dataset:

Imágenes: Almacenadas en subcarpetas según el paciente.

## Resultados
### Matriz de Confusión


|                  | Predicción: Positivo | Predicción: Negativo |
|------------------|-----------------------|-----------------------|
| **Real: Positivo** | Verdadero Positivo (TP) | Falso Negativo (FN)   |
| **Real: Negativo** | Falso Positivo (FP)     | Verdadero Negativo (TN) |

Donde:
- **TP** (True Positive): Predicciones correctas donde el modelo identificó correctamente los casos positivos.
- **FN** (False Negative): Casos positivos que el modelo identificó incorrectamente como negativos.
- **FP** (False Positive): Casos negativos que el modelo identificó incorrectamente como positivos.
- **TN** (True Negative): Predicciones correctas donde el modelo identificó correctamente los casos negativos.


![Matriz de confusion](https://github.com/melicirop/MCDA-Proyecto-20242-Deteccion_Cancer_Seno/blob/main/matriz_confusion.png)

Métrica F1-Score
Resalta los valores obtenidos durante el entrenamiento y la validación.

### Métricas derivadas de la Matriz de Confusión

1. **Precisión (Precision):**

![Precisión](https://latex.codecogs.com/png.latex?%5Ctext%7BPrecision%7D%20%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FP%7D)

   - **TP**: Verdaderos Positivos
   - **FP**: Falsos Positivos


2. **Exhaustividad (Recall o Sensitivity):**

![Recall](https://latex.codecogs.com/png.latex?%5Ctext%7BRecall%7D%20%3D%20%5Cfrac%7BTP%7D%7BTP%20&plus;%20FN%7D)

   - **TP**: Verdaderos Positivos
   - **FN**: Falsos Negativos


3. **F1-Score:**

![F1-Score](https://latex.codecogs.com/png.latex?F1%20%3D%202%20%5Ctimes%20%5Cfrac%7B%5Ctext%7BPrecision%7D%20%5Ctimes%20%5Ctext%7BRecall%7D%7D%7B%5Ctext%7BPrecision%7D%20&plus;%20%5Ctext%7BRecall%7D%7D)

   El F1-Score es una métrica que combina precisión y exhaustividad en un solo valor armónico, siendo especialmente útil cuando los datos están desbalanceados.

Resultados del proceso:

* Test Loss: 0.0291, Test F1: 0.9845

![Despliegue en gradio](https://github.com/melicirop/MCDA-Proyecto-20242-Deteccion_Cancer_Seno/blob/main/Despliegue.png)

## Autores
* Mateo Holguin Carvalho
* Melissa Andrea Ciro
* Fabian Sanchez Martinez
* Jorge Zapata Posada