import streamlit as st
import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pytube import YouTube

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="🎥 Gemini Video Assistant", page_icon="♊")

# --- 2. BARRA LATERAL PARA LA GOOGLE API KEY ---
with st.sidebar:
    st.title("🔑 Configuración Google")
    st.markdown("Introduce tu **Google AI API Key** (de Google AI Studio).")
    
    user_api_key = st.text_input("Google API Key:", type="password")
    
    if user_api_key:
        # Para Google, la variable de entorno suele llamarse GOOGLE_API_KEY
        os.environ["GOOGLE_API_KEY"] = user_api_key
        st.success("API Key de Google lista.")
    else:
        st.warning("⚠️ Esperando la clave de Google...")
        st.info("Consíguela gratis en: aistudio.google.com")

    st.divider()
    url = st.text_input("Enlace de YouTube:", placeholder="https://www.youtube.com/watch?v=...")
    btn_procesar = st.button("🚀 Analizar Video con Gemini")

if not user_api_key:
    st.info("Introduce tu Google API Key a la izquierda para activar a Gemini 👈")
    st.stop()

# --- 3. FUNCIONES DE PROCESAMIENTO (ADAPTADAS A GOOGLE) ---
@st.cache_resource(show_spinner=False)
def process_video(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    # Usamos los Embeddings de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    yt = YouTube(url)
    return vectorstore, yt.title

def stream_text_effect(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)

# --- 4. LÓGICA PRINCIPAL ---
st.title("🎥 Asistente de Video con Gemini")

if btn_procesar and url:
    try:
        with st.spinner("Gemini está leyendo el video..."):
            vs, titulo = process_video(url)
            st.session_state["vectorstore"] = vs
            st.session_state["video_title"] = titulo
            st.session_state["current_url"] = url
            st.session_state["messages"] = [] 
            st.success(f"Analizado: {titulo}")
    except Exception as e:
        st.error(f"Error: {e}")

# --- 5. CHAT ---
if "current_url" in st.session_state:
    col_vid, col_chat = st.columns([1, 1])
    
    with col_vid:
        st.video(st.session_state["current_url"])
        st.caption(f"📺 {st.session_state['video_title']}")

    with col_chat:
        if "messages" in st.session_state:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        if pregunta := st.chat_input("¿Qué quieres preguntarle a Gemini sobre el video?"):
            st.session_state.messages.append({"role": "user", "content": pregunta})
            with st.chat_message("user"):
                st.markdown(pregunta)

            with st.chat_message("assistant"):
                # Usamos el modelo Gemini 1.5 Flash (rápido y eficiente)
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state["vectorstore"].as_retriever()
                )
                
                respuesta_cruda = qa_chain.run(pregunta)
                respuesta_final = st.write_stream(stream_text_effect(respuesta_cruda))
                st.session_state.messages.append({"role": "assistant", "content": respuesta_final})