import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

try:
    from cv2.ximgproc import guidedFilter
except ImportError as e:
    print(f"Error: {e}")
    print("Necesitas instalar opencv-contrib-python manualmente.")
    print("Ejecuta: pip install opencv-contrib-python")
    exit(1)

INPUT_FOLDER = r"D:\fotosAuto"
OUTPUT_FOLDER = r"D:\fotosMejoradas"

# Parámetros ajustables
DIAMETER = 5
SIGMA_COLOR = 8
SIGMA_SPACE = 8
RADIUS = 4
EPS = 16
BILATERAL_ITERATIONS = 64
GUIDED_ITERATIONS = 4

def clean_image(
    img,
    diameter=DIAMETER,
    sigma_color=SIGMA_COLOR,
    sigma_space=SIGMA_SPACE,
    radius=RADIUS,
    eps=EPS,
    bilateral_iterations=BILATERAL_ITERATIONS,
    guided_iterations=GUIDED_ITERATIONS
):
    img = np.array(img).astype(np.float32)
    y = img.copy()

    for _ in range(bilateral_iterations):
        y = cv2.bilateralFilter(y, diameter, sigma_color, sigma_space)

    for _ in range(guided_iterations):
        y = guidedFilter(img, y, radius, eps)

    return Image.fromarray(y.clip(0, 255).astype(np.uint8))

def process_image(args):
    input_path, output_path = args
    try:
        img = Image.open(input_path)
        cleaned_image = clean_image(img)
        cleaned_image.save(output_path)
        print(f"Imagen procesada guardada: {output_path}")
        return True
    except Exception as e:
        print(f"Error procesando {input_path}: {str(e)}")
        return False

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    image_files = []
    for root, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, INPUT_FOLDER)
                output_path = os.path.join(OUTPUT_FOLDER, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image_files.append((input_path, output_path))

    print(f"Se encontraron {len(image_files)} imágenes en total.")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_image, args) for args in image_files]
        
        processed_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando imágenes"):
            if future.result():
                processed_count += 1

    print(f"Proceso completado. {processed_count} imágenes procesadas de un total de {len(image_files)}.")

if __name__ == "__main__":
    main()