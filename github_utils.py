import base64
import requests

def obtener_equivalencias_csv(repo, archivo, token):
    """
    Descarga el archivo equivalencias.csv desde GitHub y lo convierte a un diccionario.
    """
    url = f"https://api.github.com/repos/{repo}/contents/{archivo}"
    headers = {"Authorization": f"token {token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        content = base64.b64decode(response.json()['content']).decode()
        lineas = content.strip().split('\n')
        pares = [line.split(',') for line in lineas[1:] if ',' in line]
        return {color.strip(): norm.strip() for color, norm in pares}
    return {}

def subir_equivalencias_actualizadas(repo, archivo, token, equivalencias_completas):
    """
    Sube el archivo de equivalencias entero con todos los pares (existentes + nuevos).
    """
    url = f"https://api.github.com/repos/{repo}/contents/{archivo}"
    headers = {"Authorization": f"token {token}"}
    
    contenido_nuevo = "color_original,color_normalizado\n" + "\n".join(
        f"{k},{v}" for k, v in equivalencias_completas.items()
    )

    # Ver si ya existe el archivo para obtener su SHA
    response = requests.get(url, headers=headers)
    sha = response.json()["sha"] if response.status_code == 200 else None

    payload = {
        "message": "Actualizar equivalencias en bloque",
        "content": base64.b64encode(contenido_nuevo.encode()).decode(),
        "branch": "main"
    }
    if sha:
        payload["sha"] = sha

    return requests.put(url, headers=headers, json=payload)
