import requests
import json
from datetime import datetime

# Configuración
PHOTOPRISM_URL = "localhost:2342"
USERNAME = "admin"
PASSWORD = "admin"
QUALITY_THRESHOLD = 3

def get_access_token():
    print("Obteniendo token de acceso...")
    login_data = {"username": USERNAME, "password": PASSWORD}
    response = requests.post(f"{PHOTOPRISM_URL}/api/v1/session", json=login_data)
    if response.status_code == 200:
        return response.json().get('id')
    else:
        print(f"Error al obtener token. Código: {response.status_code}, Respuesta: {response.text}")
        return None

def get_photos(token, batch_size=100):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"count": batch_size, "offset": 0}
    all_photos = []

    while True:
        print(f"Obteniendo fotos con offset {params['offset']}...")
        response = requests.get(f"{PHOTOPRISM_URL}/api/v1/photos", headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error al obtener fotos. Código: {response.status_code}, Respuesta: {response.text}")
            break
        
        photos = response.json()
        if not photos:
            break
        
        all_photos.extend(photos)
        params['offset'] += len(photos)
        
        if len(photos) < batch_size:
            break

    return all_photos

def analyze_photos(all_photos):
    total_photos = len(all_photos)
    low_quality_photos = [p for p in all_photos if p.get('Quality', 5) <= QUALITY_THRESHOLD]
    low_quality_count = len(low_quality_photos)
    
    print(f"\nAnálisis de fotos:")
    print(f"Total de fotos: {total_photos}")
    print(f"Fotos de baja calidad: {low_quality_count} ({low_quality_count/total_photos*100:.2f}%)")
    
    # Análisis de calidad
    quality_distribution = {}
    for photo in all_photos:
        quality = photo.get('Quality', 0)
        quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
    
    print("\nDistribución de calidad:")
    for quality in sorted(quality_distribution.keys()):
        count = quality_distribution[quality]
        print(f"Calidad {quality}: {count} fotos ({count/total_photos*100:.2f}%)")

    # Análisis de tipos de archivo para fotos de baja calidad
    file_types = {}
    for photo in low_quality_photos:
        file_type = photo.get('Type', 'Unknown')
        file_types[file_type] = file_types.get(file_type, 0) + 1
    
    print("\nTipos de archivo de fotos de baja calidad:")
    for file_type, count in file_types.items():
        print(f"{file_type}: {count} fotos ({count/low_quality_count*100:.2f}%)")

    return low_quality_photos

def save_low_quality_photos(low_quality_photos):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"low_quality_photos_{timestamp}.json"
    
    simplified_data = [{"FileName": photo["FileName"]} for photo in low_quality_photos]
    
    with open(filename, 'w') as f:
        json.dump(simplified_data, f, indent=2)
    
    print(f"\nLa información simplificada de las fotos de baja calidad se ha guardado en '{filename}'")

def main():
    try:
        token = get_access_token()
        if token:
            print(f"Token obtenido: {token[:10]}...")
            all_photos = get_photos(token)
            low_quality_photos = analyze_photos(all_photos)
            save_low_quality_photos(low_quality_photos)
        else:
            print("No se pudo obtener el token de acceso. Verifica tus credenciales y la configuración de PhotoPrism.")
    except Exception as e:
        print(f"Se produjo un error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()