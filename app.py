import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Gemini Video AI")

with st.sidebar:
    st.title("🔑 Conexión")
    user_api_key = st.text_input("Introduce tu Google API Key:", type="password")
    if user_api_key:
        os.environ["GOOGLE_API_KEY"] = user_api_key
    
    st.divider()
    url_video = st.text_input("Link de YouTube:", placeholder="https://www.youtube.com/watch?v=...")
    btn_procesar = st.button("🚀 Analizar Video")

if not user_api_key:
    st.info("👈 Introduce tu API Key de Google en la izquierda.")
    st.stop()

# --- PROCESAMIENTO SIN PYTUBE ---
@st.cache_resource
def procesar_video(video_url):
    # Usamos el loader directo que es más estable
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    docs = loader.load()
    
    if not docs:
        raise ValueError("No se pudo obtener la transcripción. ¿El video tiene subtítulos?")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- INTERFAZ ---
st.title("🎥 Asistente de Video con Gemini")

if btn_procesar and url_video:
    try:
        with st.spinner("Analizando contenido del video..."):
            vs = procesar_video(url_video)
            st.session_state["vs"] = vs
            st.session_state["url"] = url_video
            st.session_state["chat_history"] = []
            st.success("¡Video analizado correctamente!")
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Nota: Asegúrate de que el video sea público y tenga subtítulos generados.")

if "vs" in st.session_state:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.video(st.session_state["url"])
    with col2:
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("¿Qué quieres saber?"):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                
                # Definimos el prompt del sistema
                prompt_template = ChatPromptTemplate.from_template("""
                Responde la pregunta basándote únicamente en el contexto del video proporcionado:
                <contexto>
                {context}
                </contexto>
                Pregunta: {input}""")

                combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
                retrieval_chain = create_retrieval_chain(st.session_state["vs"].as_retriever(), combine_docs_chain)
                
                response = retrieval_chain.invoke({"input": prompt})
                respuesta = response["answer"]
                st.write(respuesta)
                st.session_state["chat_history"].append({"role": "assistant", "content": respuesta})