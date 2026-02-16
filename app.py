import streamlit as st
import os
import time

# Importaciones modernas para evitar el ModuleNotFoundError
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains.retrieval_qa.base import RetrievalQA # Ruta ultra-específica
except ModuleNotFoundError as e:
    st.error(f"Error de módulos: {e}. Revisa que el requirements.txt esté correcto.")
    st.stop()

from pytube import YouTube

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Gemini Video AI")

with st.sidebar:
    st.title("🔑 Conexión")
    user_key = st.text_input("Introduce tu Google API Key:", type="password")
    if user_key:
        os.environ["GOOGLE_API_KEY"] = user_key
    
    st.divider()
    url_input = st.text_input("Link de YouTube:")
    btn = st.button("🚀 Analizar Video")

if not user_key:
    st.info("👈 Pon tu clave de Google a la izquierda.")
    st.stop()

# --- PROCESAMIENTO ---
@st.cache_resource
def procesar_video(v_url):
    loader = YoutubeLoader.from_youtube_url(v_url, add_video_info=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, YouTube(v_url).title

# --- UI ---
st.title("🎥 Asistente de Video con Gemini")

if btn and url_input:
    try:
        vs, titulo = procesar_video(url_input)
        st.session_state["vs"] = vs
        st.session_state["url"] = url_input
        st.session_state["chat_history"] = []
        st.success(f"Analizado: {titulo}")
    except Exception as e:
        st.error(f"Error: {e}")

if "vs" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.video(st.session_state["url"])
    with col2:
        for m in st.session_state.get("chat_history", []):
            with st.chat_message(m["role"]):
                st.write(m["content"])
        
        if prompt := st.chat_input("Pregunta algo..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                # Usamos la cadena de recuperación
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state["vs"].as_retriever())
                res = qa.invoke(prompt)["result"]
                st.write(res)
                st.session_state["chat_history"].append({"role": "assistant", "content": res})