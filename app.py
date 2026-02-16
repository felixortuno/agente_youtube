import streamlit as st
import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pytube import YouTube

# --- UI ---
st.set_page_config(layout="wide", page_title="Gemini Video Assistant")

with st.sidebar:
    st.title("🔑 Conexión")
    user_api_key = st.text_input("Introduce tu Google API Key:", type="password")
    if user_api_key:
        os.environ["GOOGLE_API_KEY"] = user_api_key
    
    st.divider()
    url = st.text_input("Link de YouTube:")
    btn = st.button("🚀 Analizar Video")

if not user_api_key:
    st.warning("Escribe tu Google API Key en la barra lateral 👈")
    st.stop()

# --- Lógica de Procesamiento ---
@st.cache_resource
def procesar(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Usamos Embeddings de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore, YouTube(video_url).title

# --- Ejecución ---
if btn and url:
    try:
        vs, titulo = procesar(url)
        st.session_state["vs"] = vs
        st.session_state["titulo"] = titulo
        st.session_state["url"] = url
        st.session_state["chat"] = []
        st.success(f"Analizado: {titulo}")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Interfaz de Chat ---
if "vs" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.video(st.session_state["url"])
    with col2:
        for m in st.session_state.get("chat", []):
            with st.chat_message(m["role"]):
                st.write(m["content"])
        
        if p := st.chat_input("Pregúntale al video..."):
            st.session_state["chat"].append({"role": "user", "content": p})
            with st.chat_message("user"):
                st.write(p)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state["vs"].as_retriever())
                res = qa.invoke(p)["result"]
                st.write(res)
                st.session_state["chat"].append({"role": "assistant", "content": res})