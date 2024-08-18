import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import io

try:
    from cv2.ximgproc import guidedFilter
except ImportError as e:
    print(f"Error: {e}")
    print("Necesitas instalar opencv-contrib-python manualmente.")
    print("Ejecuta: pip install opencv-contrib-python")
    exit(1)

INPUT_FOLDER = r"D:\fotosAuto"
OUTPUT_FOLDER = r"D:\fotosMejoradas"



def process_image(args):
    input_path, output_path = args
    try:
        # Leer la imagen con PIL
        with Image.open(input_path) as pil_img:
            print(f"Procesando imagen: {input_path}")
            print(f"Modo de la imagen PIL: {pil_img.mode}")
            
            # Convertir a RGB si es necesario
            if pil_img.mode != 'RGB':
                print(f"Convirtiendo imagen de {pil_img.mode} a RGB")
                pil_img = pil_img.convert('RGB')
            
            # Convertir de PIL a numpy array
            img = np.array(pil_img)
            print(f"Forma de la imagen numpy: {img.shape}")
            print(f"Tipo de datos de la imagen numpy: {img.dtype}")

        # Verificar que la imagen tiene 3 canales
        if len(img.shape) != 3 or img.shape[2] != 3:
            print(f"Error: La imagen {input_path} no tiene 3 canales. Shape: {img.shape}")
            return False

        # Redimensionar si las dimensiones son impares
        h, w = img.shape[:2]
        if h % 2 != 0 or w % 2 != 0:
            new_h = h - 1 if h % 2 != 0 else h
            new_w = w - 1 if w % 2 != 0 else w
            img = cv2.resize(img, (new_w, new_h))
            print(f"Imagen redimensionada a {img.shape}")

        # Analizar y limpiar la imagen
        params = analyze_image(img)
        cleaned_image = clean_image(img, params)
        
        # Guardar la imagen procesada
        Image.fromarray(cleaned_image).save(output_path)
        print(f"Imagen procesada guardada: {output_path}")
        print(f"Parámetros usados: {params}")
        return True
    except Exception as e:
        print(f"Error procesando {input_path}: {str(e)}")
        return False

def analyze_image(img):
    """Analiza la imagen y determina los parámetros óptimos basados en valores predeterminados."""
    print("Analizando imagen")
    print(f"Forma de la imagen en analyze_image: {img.shape}")
    print(f"Tipo de datos de la imagen en analyze_image: {img.dtype}")

    # Valores predeterminados
    DEFAULT_DIAMETER = 5
    DEFAULT_SIGMA_COLOR = 8
    DEFAULT_SIGMA_SPACE = 8
    DEFAULT_RADIUS = 4
    DEFAULT_EPS = 16
    DEFAULT_BILATERAL_ITERATIONS = 64
    DEFAULT_GUIDED_ITERATIONS = 4

    # Convertir a escala de grises usando NumPy
    gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

    # Estimar el nivel de ruido de manera más conservadora
    noise_sigma = np.std(gray) / 25  # Reducido aún más para ser más conservador

    # Calcular la complejidad de la imagen (bordes)
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    edges = np.sqrt(dx**2 + dy**2)
    edge_density = np.sum(edges > 20) / (edges.shape[0] * edges.shape[1])  # Umbral reducido aún más

    # Ajustar parámetros basados en el análisis, pero más cerca de los valores predeterminados
    diameter = int(max(3, min(DEFAULT_DIAMETER + 1, DEFAULT_DIAMETER + noise_sigma * 2)))
    sigma_color = max(DEFAULT_SIGMA_COLOR - 2, min(DEFAULT_SIGMA_COLOR + 2, DEFAULT_SIGMA_COLOR + noise_sigma * 5))
    sigma_space = max(DEFAULT_SIGMA_SPACE - 2, min(DEFAULT_SIGMA_SPACE + 2, DEFAULT_SIGMA_SPACE + noise_sigma * 5))
    radius = int(max(DEFAULT_RADIUS - 1, min(DEFAULT_RADIUS + 1, DEFAULT_RADIUS + edge_density * 2)))
    eps = max(DEFAULT_EPS / 2, min(DEFAULT_EPS * 1.5, DEFAULT_EPS + noise_sigma * 20))
    bilateral_iterations = int(max(DEFAULT_BILATERAL_ITERATIONS / 2, min(DEFAULT_BILATERAL_ITERATIONS * 1.5, DEFAULT_BILATERAL_ITERATIONS + noise_sigma * 50)))
    guided_iterations = int(max(DEFAULT_GUIDED_ITERATIONS - 1, min(DEFAULT_GUIDED_ITERATIONS + 1, DEFAULT_GUIDED_ITERATIONS + noise_sigma * 2)))

    return {
        'diameter': diameter,
        'sigma_color': sigma_color,
        'sigma_space': sigma_space,
        'radius': radius,
        'eps': eps,
        'bilateral_iterations': bilateral_iterations,
        'guided_iterations': guided_iterations
    }

def clean_image(img, params):
    """Limpia la imagen usando filtros bilaterales y guiados con ajustes para preservar texturas."""
    print("Limpiando imagen")
    img = img.astype(np.float32) / 255.0  # Normalizar a [0, 1]

    # Aplicar filtro bilateral con parámetros más suaves
    bilateral = cv2.bilateralFilter(img, params['diameter'], 
                                    params['sigma_color'] / 2, 
                                    params['sigma_space'] / 2)

    # Aplicar filtro guiado con parámetros ajustados
    guided = cv2.ximgproc.guidedFilter(img, bilateral, 
                                       params['radius'], 
                                       params['eps'] / 5)

    # Mezclar la imagen original con la imagen filtrada
    alpha = 0.3  # Reducido de 0.7 a 0.3 para una mezcla más equilibrada
    result = alpha * guided + (1 - alpha) * img

    return (result * 255).clip(0, 255).astype(np.uint8)

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
