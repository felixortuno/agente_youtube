import streamlit as st
import time
import os
from langchain_openai import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pytube import YouTube

# --- 1. CONFIGURACIÓN DE PÁGINA Y API ---
st.set_page_config(layout="wide", page_title="🎥 AI Video Assistant", page_icon="🤖")

# Puedes poner tu clave aquí directamente para probar (no recomendado para subir a GitHub)
# os.environ["OPENAI_API_KEY"] = "tu-clave-aquí"

if "OPENAI_API_KEY" not in os.environ:
    st.error("⚠️ Falta la clave OPENAI_API_KEY. Configúrala en tus variables de entorno.")
    st.stop()

# --- 2. FUNCIONES DE PROCESAMIENTO ---
@st.cache_resource(show_spinner=False)
def process_video(url):
    """Carga el video, lo divide en trozos y crea la base de datos de conocimiento"""
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    docs = loader.load()
    
    # Dividimos el texto para no exceder los límites de la IA
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # Creamos la base de datos vectorial (el "cerebro" del agente)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Obtenemos el título real del video
    yt = YouTube(url)
    return vectorstore, yt.title

def stream_text_effect(text):
    """Genera el efecto de escritura palabra por palabra"""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

# --- 3. INTERFAZ DE USUARIO (UI) ---
st.title("🎥 Asistente Multimedia Inteligente")
st.write("Analiza videos de YouTube y chatea con ellos en tiempo real.")

with st.sidebar:
    st.header("Configuración")
    url = st.text_input("Enlace de YouTube:", placeholder="https://www.youtube.com/watch?v=...")
    btn_procesar = st.button("🚀 Analizar Video")

# --- 4. LÓGICA DE PROCESAMIENTO ---
if btn_procesar and url:
    try:
        with st.spinner("Leyendo la transcripción del video..."):
            vs, titulo = process_video(url)
            st.session_state["vectorstore"] = vs
            st.session_state["video_title"] = titulo
            st.session_state["current_url"] = url
            st.session_state["messages"] = [] # Limpiar chat al cambiar de video
            st.success(f"Listo: {titulo}")
    except Exception as e:
        st.error(f"Error: {e}")

# --- 5. ÁREA DE CHAT Y VIDEO ---
col_vid, col_chat = st.columns([1, 1])

if "current_url" in st.session_state:
    with col_vid:
        st.video(st.session_state["current_url"])
        st.caption(f"📺 {st.session_state['video_title']}")

    with col_chat:
        # Mostrar historial de chat
        if "messages" in st.session_state:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        # Entrada del chat
        if pregunta := st.chat_input("¿Qué quieres saber del video?"):
            st.session_state.messages.append({"role": "user", "content": pregunta})
            with st.chat_message("user"):
                st.markdown(pregunta)

            # Generar respuesta
            with st.chat_message("assistant"):
                llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state["vectorstore"].as_retriever()
                )
                
                with st.spinner("Consultando al video..."):
                    respuesta_cruda = qa_chain.run(pregunta)
                    # AQUÍ ESTÁ EL CAMBIO: Mostramos con efecto streaming
                    respuesta_final = st.write_stream(stream_text_effect(respuesta_cruda))
                
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})
else:
    st.info("👈 Pega un enlace de YouTube en la barra lateral para comenzar.")