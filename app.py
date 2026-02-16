import streamlit as st
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pytube import YouTube

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Gemini Video AI", page_icon="♊")

with st.sidebar:
    st.title("🔑 Conexión")
    user_api_key = st.text_input("Introduce tu Google API Key:", type="password")
    if user_api_key:
        os.environ["GOOGLE_API_KEY"] = user_api_key
    
    st.divider()
    url_video = st.text_input("Link de YouTube:", placeholder="https://www.youtube.com/watch?v=...")
    btn_procesar = st.button("🚀 Analizar Video")

if not user_api_key:
    st.info("👈 Introduce tu API Key de Google en la izquierda para empezar.")
    st.stop()

# --- PROCESAMIENTO ---
@st.cache_resource
def procesar_video(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Embeddings de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore, YouTube(video_url).title

# --- INTERFAZ ---
st.title("🎥 Asistente de Video con Gemini")

if btn_procesar and url_video:
    try:
        with st.spinner("Analizando contenido..."):
            vs, titulo = procesar_video(url_video)
            st.session_state["vs"] = vs
            st.session_state["titulo"] = titulo
            st.session_state["url"] = url_video
            st.session_state["chat_history"] = []
            st.success(f"Analizado: {titulo}")
    except Exception as e:
        st.error(f"Error: {e}")

if "vs" in st.session_state:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.video(st.session_state["url"])
    
    with col2:
        # Contenedor para el chat
        chat_container = st.container()
        
        for msg in st.session_state["chat_history"]:
            with chat_container.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("¿Qué quieres saber?"):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with chat_container.chat_message("user"):
                st.write(prompt)
            
            with chat_container.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                qa = RetrievalQA.from_chain_type(
                    llm=llm, 
                    retriever=st.session_state["vs"].as_retriever()
                )
                
                # Usamos invoke para evitar avisos de depreciación
                resultado = qa.invoke(prompt)
                respuesta = resultado["result"]
                st.write(respuesta)
                
            st.session_state["chat_history"].append({"role": "assistant", "content": respuesta})