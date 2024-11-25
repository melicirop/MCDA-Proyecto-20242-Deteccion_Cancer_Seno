import numpy as np
import os
from tqdm import tqdm  # Importar la versión de tqdm para notebooks
from pathlib import Path
import dicomsdl
import multiprocessing as mp
from PIL import Image

RESIZE_TO = (512, 512)

def dicom_file_to_ary(path):
    dcm_file = dicomsdl.open(str(path))
    data = dcm_file.pixelData()

    # Convertir explícitamente a un array de NumPy y verificar el tipo
    data = np.array(data, dtype=np.float32)

    if data.size == 0 or not isinstance(data, np.ndarray):
        raise ValueError(f"La imagen en {path} no contiene datos válidos para procesar o no es un array de NumPy.")

    # Verificar si el array es continuo en memoria
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    # Asegurarse de que data sea 2D
    if len(data.shape) != 2:
        raise ValueError(f"La imagen en {path} no es 2D y no se puede redimensionar con PIL. Forma actual: {data.shape}")

    # Normalizar y convertir a uint8 antes de redimensionar
    data = (data - data.min()) / (data.max() - data.min())
    data = (data * 255).astype(np.uint8)

    # Usar PIL para redimensionar
    try:
        image = Image.fromarray(data)
        image = image.resize(RESIZE_TO, Image.LANCZOS)
        data_resized = np.array(image, dtype=np.uint8)  # Convertir explícitamente a uint8
        data_resized = np.ascontiguousarray(data_resized)

    except Exception as e:
        print(f"Error al redimensionar la imagen en {path} con PIL: {e}")
        raise

    return data_resized

def process_directory(directory_path):
    parent_directory = str(directory_path).split('/')[-1]
    output_dir = f'train_images_processed_cv2_dicomsdl_{RESIZE_TO[0]}/{parent_directory}'
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = list(directory_path.iterdir())
    
    with tqdm(total=len(image_paths), desc=f'Procesando carpeta {parent_directory}', leave=False) as pbar:
        for image_path in image_paths:
            processed_ary = dicom_file_to_ary(image_path)
            
            # Guardar la imagen usando PIL
            image_to_save = Image.fromarray(processed_ary)
            image_to_save.save(f'{output_dir}/{image_path.stem}.png')
            
            pbar.update(1)

directories = list(Path('/kaggle/input/rsna-breast-cancer-detection/train_images').iterdir())

with mp.Pool(mp.cpu_count()) as pool:
    for _ in tqdm(pool.imap_unordered(process_directory, directories), total=len(directories), desc="Progreso general de carpetas"):
        pass