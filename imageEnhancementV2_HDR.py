import cv2
import numpy as np
from PIL import Image, ImageEnhance
from skimage import restoration, exposure
from scipy.ndimage import gaussian_filter1d
import os
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def estimate_noise(image):
    try:
        return np.mean(restoration.estimate_sigma(image, channel_axis=-1))
    except TypeError:
        return np.mean(restoration.estimate_sigma(image, multichannel=True))

def advanced_debanding(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    
    median_filtered = cv2.medianBlur(l_channel, 5)
    diff = l_channel.astype(np.float32) - median_filtered.astype(np.float32)
    
    threshold = np.std(diff) * 2
    mask = np.abs(diff) > threshold
    
    smoothed = gaussian_filter1d(l_channel, sigma=3, axis=0)
    l_deband = np.where(mask, smoothed, l_channel)
    
    lab[:,:,0] = l_deband
    deband_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return deband_image

def enhance_image(input_path, output_path):
    logging.info(f"Iniciando proceso de mejora de imagen: {input_path}")
    
    img_pil = Image.open(input_path)
    img_rgb = np.array(img_pil)
    
    noise_level = estimate_noise(img_rgb)
    logging.info(f"Nivel de ruido estimado: {noise_level}")
    
    try:
        img_denoised = restoration.denoise_wavelet(img_rgb, channel_axis=-1, 
                                                   method='BayesShrink', mode='soft',
                                                   rescale_sigma=True)
    except TypeError:
        img_denoised = restoration.denoise_wavelet(img_rgb, multichannel=True, 
                                                   method='BayesShrink', mode='soft',
                                                   rescale_sigma=True)
    
    img_denoised = (img_denoised * 255).astype(np.uint8)
    
    img_debanded = advanced_debanding(img_denoised)
    
    # Ajustar el efecto HDR-like
    img_adjusted = exposure.equalize_adapthist(img_debanded, clip_limit=0.005, kernel_size=16)  # kernel =  8, 16, o 32
    
    
    img_adjusted = exposure.rescale_intensity(img_adjusted, in_range=(0.05, 0.95))
    
    pil_img = Image.fromarray((img_adjusted * 255).astype('uint8'))
    
    
    enhancer = ImageEnhance.Color(pil_img)
    pil_img = enhancer.enhance(1.05)
    
    
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    pil_img.save(output_path, quality=95)
    logging.info(f"Imagen mejorada guardada en: {output_path}")

def process_images(input_folder, output_folder, json_file):
    os.makedirs(output_folder, exist_ok=True)
    
    with open(json_file, 'r') as f:
        image_data = json.load(f)
    
    error_report = []
    
    for img_info in tqdm(image_data, desc="Processing images"):
        normalized_filename = img_info['FileName'].replace('/', os.path.sep)
        input_path = os.path.join(input_folder, normalized_filename)
        output_path = os.path.join(output_folder, normalized_filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Procesando: {input_path}")
        
        if os.path.exists(input_path):
            try:
                enhance_image(input_path, output_path)
            except Exception as e:
                error_msg = f"Error procesando {normalized_filename}: {str(e)}"
                print(error_msg)
                error_report.append(error_msg)
        else:
            error_msg = f"Archivo no encontrado: {input_path}"
            print(error_msg)
            error_report.append(error_msg)
    
    if error_report:
        with open(os.path.join(output_folder, 'error_report.txt'), 'w') as f:
            for error in error_report:
                f.write(f"{error}\n")
        print(f"Se ha generado un informe de errores en {os.path.join(output_folder, 'error_report.txt')}")
    else:
        print("Todas las imágenes fueron procesadas exitosamente.")

if __name__ == "__main__":
    input_folder = r"D:\fotosMalas\3"
    output_folder = r"D:\fotosReparadasHDR"
    json_file = "low.json"
    process_images(input_folder, output_folder, json_file)