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
st.set_page_config(layout="wide", page_title="Gemini Video AI")

with st.sidebar:
    st.title("🔑 Configuración")
    user_api_key = st.text_input("Introduce tu Google API Key:", type="password")
    if user_api_key:
        os.environ["GOOGLE_API_KEY"] = user_api_key
    
    st.divider()
    url = st.text_input("Link de YouTube:")
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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, YouTube(video_url).title

# --- INTERFAZ PRINCIPAL ---
st.title("🎥 Asistente de Video (Gemini)")

if btn_procesar and url:
    try:
        vs, titulo = procesar_video(url)
        st.session_state["vs"] = vs
        st.session_state["titulo"] = titulo
        st.session_state["url"] = url
        st.session_state["chat_history"] = []
        st.success(f"Analizado: {titulo}")
    except Exception as e:
        st.error(f"Error: {e}")

if "vs" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.video(st.session_state["url"])
    with col2:
        # Mostrar historial
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Nuevo mensaje
        if prompt := st.chat_input("Pregunta algo sobre el video..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state["vs"].as_retriever())
                respuesta = qa.invoke(prompt)["result"]
                st.write(respuesta)
                st.session_state["chat_history"].append({"role": "assistant", "content": respuesta})