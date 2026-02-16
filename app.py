import streamlit as st
import os

# --- Importaciones seguras para la versión 0.1.20 ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    # En la versión 0.1.20, esta importación FUNCIONA SIEMPRE
    from langchain.chains import RetrievalQA
except ImportError as e:
    st.error(f"Error CRÍTICO de importación: {e}")
    st.stop()

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gemini Video AI", layout="wide")

with st.sidebar:
    st.title("🔒 Conexión Segura")
    # AQUÍ es donde pegarás tu clave nueva cuando la app arranque
    api_key = st.text_input("Pega tu NUEVA Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    url = st.text_input("Link de YouTube:")
    btn = st.button("Analizar Video")

if not api_key:
    st.warning("👈 Pega tu nueva API Key en la barra lateral para empezar.")
    st.stop()

# --- PROCESAMIENTO ---
@st.cache_resource
def procesar_video(video_url):
    # add_video_info=False evita errores de bloqueo de YouTube
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False, language=["es", "en"])
    docs = loader.load()
    
    if not docs:
        raise ValueError("No se encontraron subtítulos en este video.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# --- INTERFAZ ---
st.title("🤖 Chat con Video (Gemini)")

if btn and url:
    try:
        with st.spinner("Procesando video..."):
            st.session_state.vs = procesar_video(url)
            st.session_state.url = url
            st.success("¡Listo! Pregunta lo que quieras.")
    except Exception as e:
        st.error(f"Ocurrió un error: {e}")

if "vs" in st.session_state:
    c1, c2 = st.columns(2)
    with c1:
        st.video(st.session_state.url)
    with c2:
        if "chat" not in st.session_state:
            st.session_state.chat = []
            
        for m in st.session_state.chat:
            with st.chat_message(m["role"]): st.write(m["content"])
        
        if prompt := st.chat_input("Escribe tu pregunta..."):
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.write(prompt)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vs.as_retriever()
                )
                res = qa.invoke(prompt)
                ans = res["result"]
                st.write(ans)
                st.session_state.chat.append({"role": "assistant", "content": ans})