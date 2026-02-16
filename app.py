import streamlit as st
import os
import time

# Importaciones protegidas
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    # Usamos la importación más básica posible
    from langchain.chains import RetrievalQA
except Exception as e:
    st.error(f"Error de módulos: {e}")
    st.info("Espera a que Streamlit termine de instalar las dependencias en 'Manage app'.")
    st.stop()

from pytube import YouTube

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Gemini Video Chat")

with st.sidebar:
    st.title("🔑 Configuración")
    api_key = st.text_input("Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    url = st.text_input("Link de YouTube:")
    btn = st.button("🚀 Analizar Video")

if not api_key:
    st.warning("👈 Introduce tu API Key de Google.")
    st.stop()

# --- PROCESAMIENTO ---
@st.cache_resource
def procesar(v_url):
    loader = YoutubeLoader.from_youtube_url(v_url, add_video_info=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, YouTube(v_url).title

# --- UI ---
st.title("🎥 Chat con Video (Gemini)")

if btn and url:
    try:
        vs, titulo = procesar(url)
        st.session_state["vs"] = vs
        st.session_state["url"] = url
        st.session_state["chat"] = []
        st.success(f"Analizado: {titulo}")
    except Exception as e:
        st.error(f"Error: {e}")

if "vs" in st.session_state:
    c1, c2 = st.columns(2)
    with c1:
        st.video(st.session_state["url"])
    with c2:
        for m in st.session_state.get("chat", []):
            with st.chat_message(m["role"]): st.write(m["content"])
        
        if p := st.chat_input("Pregunta algo..."):
            st.session_state["chat"].append({"role": "user", "content": p})
            with st.chat_message("user"): st.write(p)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state["vs"].as_retriever())
                res = qa.invoke(p)["result"]
                st.write(res)
                st.session_state["chat"].append({"role": "assistant", "content": res})