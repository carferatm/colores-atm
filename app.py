import streamlit as st
import pandas as pd
import openai
import re
import numpy as np
import os
import io
import faiss
import time
from typing import List
from github_utils import obtener_equivalencias_csv, subir_equivalencias_actualizadas

st.set_page_config(page_title="Normalizador de Colores ATM", layout="wide")

with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("API Key de OpenAI", type="password")
    embedding_model = st.selectbox("Modelo de Embedding", ["text-embedding-3-small", "text-embedding-3-large"])
    if st.button("Guardar configuración"):
        if api_key:
            st.session_state.api_key = api_key
            st.session_state.embedding_model = embedding_model
            st.success("Configuración guardada")

REPO_GITHUB = os.getenv("REPO_GITHUB", st.secrets["github"]["repo"])
ARCHIVO_EQUIVALENCIAS = "equivalencias.csv"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", st.secrets["github"]["token"])

colores_atm = [
    "Rojo", "Amarillo", "Azul", "Azul Marino", "Naranja", "Verde", "Morado", "Rosa", "Blanco", "Negro",
    "Beige", "Marrón", "Gris", "Azul Royal", "Fucsia", "Granate", "Lila", "Coral", "Kaki", "Lima", "Verde Oliva",
    "Salmón", "Plata", "Dorado"
]

def crear_o_cargar_faiss(api_key: str, modelo: str):
    st.info("Cargando o generando embeddings de colores ATM...")
    if os.path.exists("embeddings_atm.npy") and os.path.exists("index.faiss"):
        emb_atm = np.load("embeddings_atm.npy")
        index = faiss.read_index("index.faiss")
    else:
        openai.api_key = api_key
        response = openai.embeddings.create(
            model=modelo,
            input=colores_atm
        )
        emb_atm = np.array([d.embedding for d in response.data]).astype("float32")
        np.save("embeddings_atm.npy", emb_atm)

        index = faiss.IndexFlatIP(emb_atm.shape[1])
        faiss.normalize_L2(emb_atm)
        index.add(emb_atm)
        faiss.write_index(index, "index.faiss")

    return emb_atm, index

def preprocesar_color(cadena: str) -> List[str]:
    colores = re.split(r"[-/,]", str(cadena))
    return [c.strip().lower() for c in colores if c.strip()]

def generar_embeddings(api_key: str, modelo: str, lista_colores: List[str], batch_size: int = 1000) -> np.ndarray:
    openai.api_key = api_key
    lista_limpia = [str(c).strip() for c in lista_colores if isinstance(c, str) or isinstance(c, (int, float))]
    if not lista_limpia:
        raise ValueError("La lista de colores a procesar está vacía.")
    embeddings = []
    for i in range(0, len(lista_limpia), batch_size):
        st.info(f"Generando embeddings del {i} al {min(i+batch_size, len(lista_limpia))}...")
        batch = lista_limpia[i:i + batch_size]
        response = openai.embeddings.create(
            model=modelo,
            input=batch
        )
        embeddings.extend([d.embedding for d in response.data])
    return np.array(embeddings)

def normalizar_colores_faiss(colores_raw: List[str], index, nombres_colores: List[str], emb_entrada: np.ndarray) -> List[str]:
    faiss.normalize_L2(emb_entrada)
    st.info(f"Buscando similitudes con {len(emb_entrada)} colores nuevos no normalizados...")
    start = time.time()
    _, indices = index.search(emb_entrada, 1)
    st.info(f"Similitud calculada en {time.time() - start:.2f} segundos")
    resultados = []
    for i, idx in enumerate(indices[:, 0]):
        if idx == -1:
            resultados.append(f"UNKNOWN({colores_raw[i]})")
        else:
            resultados.append(nombres_colores[idx])
    return resultados

st.title("Normalizador de Colores - ATMÓSFERA SPORT")
csv_file = st.file_uploader("Sube un CSV con una columna de colores", type="csv")

if csv_file and "api_key" in st.session_state:
    try:
        df = pd.read_csv(io.StringIO(csv_file.getvalue().decode("utf-8")))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(csv_file.getvalue().decode("latin1")))

    st.write("Vista previa del archivo:", df.head())

    if st.button("Buscar colores ATM"):
        with st.spinner("Procesando archivo y generando resultados..."):
            emb_atm, index = crear_o_cargar_faiss(
                st.session_state.api_key,
                st.session_state.embedding_model
            )

            st.info("Preprocesando archivo de entrada...")
            lista_colores = df.iloc[:, 0].fillna("").apply(preprocesar_color)
            colores_flat = [color for sublist in lista_colores for color in sublist]

            equivalencias = obtener_equivalencias_csv(REPO_GITHUB, ARCHIVO_EQUIVALENCIAS, GITHUB_TOKEN)

            colores_existentes = []
            colores_por_normalizar = []

            for color in colores_flat:
                if color in equivalencias:
                    colores_existentes.append(equivalencias[color])
                else:
                    colores_por_normalizar.append(color)

            colores_por_normalizar = list(set(colores_por_normalizar))

            if colores_por_normalizar:
                start_emb = time.time()
                emb_entrada = generar_embeddings(
                    st.session_state.api_key,
                    st.session_state.embedding_model,
                    colores_por_normalizar
                ).astype("float32")
                st.info(f"Embeddings generados en {time.time() - start_emb:.2f} segundos")

                nuevos_resultados = normalizar_colores_faiss(
                    colores_por_normalizar,
                    index,
                    colores_atm,
                    emb_entrada
                )

                equivalencias.update(dict(zip(colores_por_normalizar, nuevos_resultados)))
                subir_equivalencias_actualizadas(
                    REPO_GITHUB,
                    ARCHIVO_EQUIVALENCIAS,
                    GITHUB_TOKEN,
                    equivalencias
                )

                for color in colores_flat:
                    colores_existentes.append(equivalencias.get(color, f"UNKNOWN({color})"))

            idx = 0
            colores_por_fila = []
            for sublist in lista_colores:
                n = len(sublist)
                normalizados = colores_existentes[idx:idx+n]
                colores_por_fila.append("/".join(normalizados))
                idx += n

            df["colores atm"] = colores_por_fila

        st.success("Procesamiento completo")
        num_trabajados = df["colores atm"].notnull().sum()
        st.info(f"Nº total de productos trabajados: {num_trabajados}")
        st.download_button("Descargar CSV con colores ATM",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name="colores_atm.csv",
                           mime="text/csv")
        st.dataframe(df)

elif csv_file:
    st.warning("Debes ingresar y guardar la API Key de OpenAI para continuar.")
