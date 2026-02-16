import streamlit as st
import os

# Importaciones modernas y seguras
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    # Esta es la forma más estable de importar la cadena ahora
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from langchain_core.prompts import ChatPromptTemplate
except Exception as e:
    st.error(f"Error crítico de librerías: {e}")
    st.stop()

from pytube import YouTube

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Gemini Video AI")

with st.sidebar:
    st.title("🔑 Conexión")
    api_key = st.text_input("Introduce tu Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    url = st.text_input("Link de YouTube:")
    btn = st.button("🚀 Analizar Video")

if not api_key:
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

# --- UI Y CHAT ---
st.title("🎥 Asistente de Video con Gemini")

if btn and url:
    vs, titulo = procesar_video(url)
    st.session_state["vs"] = vs
    st.session_state["url"] = url
    st.session_state["chat_history"] = []

if "vs" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        st.video(st.session_state["url"])
    with col2:
        for m in st.session_state.get("chat_history", []):
            with st.chat_message(m["role"]): st.markdown(m["content"])
        
        if prompt := st.chat_input("Pregunta algo..."):
            st.session_state["chat_history"].append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                
                # Nueva forma de crear la cadena (Chain) en LangChain 0.3
                system_prompt = (
                    "Usa el siguiente contexto para responder la pregunta. "
                    "Si no sabes la respuesta, di que no lo sabes. "
                    "\n\n"
                    "{context}"
                )
                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, chat_prompt)
                rag_chain = create_retrieval_chain(st.session_state["vs"].as_retriever(), question_answer_chain)
                
                response = rag_chain.invoke({"input": prompt})
                full_res = response["answer"]
                st.markdown(full_res)
                st.session_state["chat_history"].append({"role": "assistant", "content": full_res})