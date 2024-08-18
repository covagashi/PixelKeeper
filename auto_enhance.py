from wand.image import Image
import os
import json
from tqdm import tqdm

def enhance_image(input_path, output_path):
    with Image(filename=input_path) as img:
        # Aplicar mejoras automáticas
        img.auto_level()
        img.enhance()
        img.unsharp_mask(radius=1.5, sigma=1.0, amount=1.0, threshold=0.0)
        
        # Ajuste fino de brillo y contraste
        img.brightness_contrast(brightness=5, contrast=10)
        
        # Guardar la imagen mejorada
        img.save(filename=output_path)

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
    output_folder = r"D:\fotosAuto"
    json_file = r"low.json"
    process_images(input_folder, output_folder, json_file)