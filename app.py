import streamlit as st
import os

# Importaciones protegidas para detectar fallos de instalación
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
except Exception as e:
    st.error(f"Error de módulos: {e}")
    st.info("Revisa que el archivo requirements.txt tenga las versiones correctas.")
    st.stop()

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Gemini Video AI")

with st.sidebar:
    st.title("🔑 Conexión")
    api_key = st.text_input("Introduce tu Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    url_input = st.text_input("Link de YouTube:", placeholder="https://www.youtube.com/watch?v=...")
    btn_analizar = st.button("🚀 Analizar Video")

if not api_key:
    st.info("👈 Introduce tu clave de Google AI Studio a la izquierda.")
    st.stop()

# --- PROCESAMIENTO ---
@st.cache_resource
def procesar_contenido(url):
    # add_video_info=False es lo que evita el bloqueo de YouTube
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs = loader.load()
    
    if not docs:
        raise ValueError("No se encontraron subtítulos. Prueba con otro video.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- INTERFAZ ---
st.title("🎥 Asistente de Video con Gemini")

if btn_analizar and url_input:
    try:
        with st.spinner("Gemini está analizando el video..."):
            st.session_state["vs"] = procesar_contenido(url_input)
            st.session_state["url"] = url_input
            st.session_state["chat_history"] = []
            st.success("¡Video listo para chatear!")
    except Exception as e:
        st.error(f"Error: {e}")

if "vs" in st.session_state:
    col_v, col_c = st.columns([1, 1])
    with col_v:
        st.video(st.session_state["url"])
    with col_c:
        for m in st.session_state["chat_history"]:
            with st.chat_message(m["role"]): st.write(m["content"])
        
        if p := st.chat_input("¿Qué quieres saber?"):
            st.session_state["chat_history"].append({"role": "user", "content": p})
            with st.chat_message("user"): st.write(p)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                prompt = ChatPromptTemplate.from_template("Responde basándote solo en este video: {context}\nPregunta: {input}")
                
                chain_docs = create_stuff_documents_chain(llm, prompt)
                chain_rag = create_retrieval_chain(st.session_state["vs"].as_retriever(), chain_docs)
                
                res = chain_rag.invoke({"input": p})
                ans = res["answer"]
                st.write(ans)
                st.session_state["chat_history"].append({"role": "assistant", "content": ans})