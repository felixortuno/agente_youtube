import streamlit as st
import time
import os
# Importaciones actualizadas para Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Cambio aquí
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pytube import YouTube

# --- Configuración de la App ---
st.set_page_config(layout="wide", page_title="Gemini Video AI")

with st.sidebar:
    st.title("🔑 Configuración")
    user_api_key = st.text_input("Introduce tu Google API Key:", type="password")
    if user_api_key:
        os.environ["GOOGLE_API_KEY"] = user_api_key
    
    st.divider()
    url = st.text_input("Pega el link de YouTube aquí:")
    btn = st.button("🚀 Analizar Video")

if not user_api_key:
    st.warning("👈 Por favor, introduce tu Google API Key a la izquierda.")
    st.stop()

# --- Procesamiento del Video ---
@st.cache_resource
def procesar_video(v_url):
    loader = YoutubeLoader.from_youtube_url(v_url, add_video_info=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Usamos embeddings de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore, YouTube(v_url).title

# --- Interfaz de Usuario ---
st.title("🎥 Chatea con tus videos (Gemini)")

if btn and url:
    try:
        vs, titulo = procesar_video(url)
        st.session_state["vs"] = vs
        st.session_state["titulo"] = titulo
        st.session_state["url"] = url
        st.session_state["chat_history"] = []
        st.success(f"Analizado con éxito: {titulo}")
    except Exception as e:
        st.error(f"Hubo un problema: {e}")

# --- Chat ---
if "vs" in st.session_state:
    col_v, col_c = st.columns(2)
    with col_v:
        st.video(st.session_state["url"])
    with col_c:
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("¿De qué trata el video?"):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state["vs"].as_retriever())
                # Usamos invoke en lugar de run (más moderno)
                respuesta = qa.invoke(prompt)["result"]
                st.write(respuesta)
                st.session_state["chat_history"].append({"role": "assistant", "content": respuesta})import streamlit as st
import time
import os
# Importaciones actualizadas para Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Cambio aquí
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pytube import YouTube

# --- Configuración de la App ---
st.set_page_config(layout="wide", page_title="Gemini Video AI")

with st.sidebar:
    st.title("🔑 Configuración")
    user_api_key = st.text_input("Introduce tu Google API Key:", type="password")
    if user_api_key:
        os.environ["GOOGLE_API_KEY"] = user_api_key
    
    st.divider()
    url = st.text_input("Pega el link de YouTube aquí:")
    btn = st.button("🚀 Analizar Video")

if not user_api_key:
    st.warning("👈 Por favor, introduce tu Google API Key a la izquierda.")
    st.stop()

# --- Procesamiento del Video ---
@st.cache_resource
def procesar_video(v_url):
    loader = YoutubeLoader.from_youtube_url(v_url, add_video_info=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Usamos embeddings de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore, YouTube(v_url).title

# --- Interfaz de Usuario ---
st.title("🎥 Chatea con tus videos (Gemini)")

if btn and url:
    try:
        vs, titulo = procesar_video(url)
        st.session_state["vs"] = vs
        st.session_state["titulo"] = titulo
        st.session_state["url"] = url
        st.session_state["chat_history"] = []
        st.success(f"Analizado con éxito: {titulo}")
    except Exception as e:
        st.error(f"Hubo un problema: {e}")

# --- Chat ---
if "vs" in st.session_state:
    col_v, col_c = st.columns(2)
    with col_v:
        st.video(st.session_state["url"])
    with col_c:
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("¿De qué trata el video?"):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state["vs"].as_retriever())
                # Usamos invoke en lugar de run (más moderno)
                respuesta = qa.invoke(prompt)["result"]
                st.write(respuesta)
                st.session_state["chat_history"].append({"role": "assistant", "content": respuesta})