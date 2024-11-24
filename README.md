# Detección de Cáncer en Imágenes Radiológicas con Modelos Combinados
Este proyecto implementa un modelo combinado de aprendizaje profundo y aprendizaje tabular para la detección de cáncer en imágenes radiológicas. Integra datos de imágenes y características clínicas para proporcionar predicciones más precisas.

## Descripción del Proyecto

La herramienta utiliza:
Redes neuronales convolucionales (CNN) para extraer características de imágenes.
Redes densas para procesar variables tabulares (edad, densidad mamaria, etc.).
Un modelo combinado que une ambas ramas para realizar una clasificación binaria: Cáncer o No Cáncer.
Además, se incluye una interfaz amigable desarrollada con Gradio para facilitar su uso por profesionales médicos.

## Estructura del proyecto

## Caracteristicas

* Procesamiento de Imágenes:

    * Conversión de imágenes en formato DICOM a PNG.
    * Generación de mapas de calor con el modelo VGG16 preentrenado.
    * Redimensionamiento y normalización de imágenes.

* Procesamiento de Datos Tabulares:

    * Escalado de características clínicas (edad, densidad, etc.) mediante StandardScaler.
    * Combinación con características extraídas de imágenes.

* Entrenamiento y Evaluación:

    * Modelo combinado con ResNet18 como rama de imágenes.
    * Métrica de evaluación: F1-Score.
    * Generación de matriz de confusión para evaluar el rendimiento.

* Despliegue:

    * Interfaz amigable en Gradio para subir imágenes y características clínicas.
    * Predicción de clase (Cáncer o No Cáncer) con confianza asociada.

## Tecnologías Utilizadas

Python (Librerías principales):
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
Datos Tabulares: Archivo CSV con columnas como edad, densidad, BIRADS, etc.