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

def agregar_equivalencia_a_github(repo, archivo, token, nuevo_registro):
    """
    Agrega una línea a equivalencias.csv en GitHub mediante la API.
    """
    url = f"https://api.github.com/repos/{repo}/contents/{archivo}"
    headers = {"Authorization": f"token {token}"}
    
    # Leer contenido actual
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        contenido_actual = base64.b64decode(response.json()['content']).decode()
        sha = response.json()['sha']
        nuevo_contenido = contenido_actual.strip() + f"\n{nuevo_registro}"
    else:
        sha = None
        nuevo_contenido = "color_original,color_normalizado\n" + nuevo_registro

    payload = {
        "message": f"Agregar equivalencia: {nuevo_registro}",
        "content": base64.b64encode(nuevo_contenido.encode()).decode(),
        "branch": "main",
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers, json=payload)
    return r.status_code, r.json()
