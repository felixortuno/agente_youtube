import streamlit as st
import os

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.document_loaders import YoutubeLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
except Exception as e:
    st.error(f"Error importando librerías: {e}")
    st.stop()

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gemini Video AI", layout="wide")

with st.sidebar:
    st.title("🤖 Configuración")
    api_key = st.text_input("Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    url = st.text_input("Link de YouTube:")
    btn = st.button("Analizar Video")

if not api_key:
    st.info("👈 Ingresa tu API Key para comenzar.")
    st.stop()

# --- LÓGICA ---
@st.cache_resource
def procesar_video(link):
    # Sin metadatos para evitar el bloqueo HTTP de YouTube
    loader = YoutubeLoader.from_youtube_url(link, add_video_info=False)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# --- INTERFAZ ---
st.title("🎬 Chat con Videos de YouTube")

if btn and url:
    try:
        with st.spinner("Descargando subtítulos y procesando..."):
            st.session_state.vectorstore = procesar_video(url)
            st.session_state.url = url
            st.session_state.messages = []
            st.success("¡Video procesado!")
    except Exception as e:
        st.error(f"Error al procesar: {e}")

if "vectorstore" in st.session_state:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.video(st.session_state.url)
    
    with col2:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Pregunta algo sobre el video..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Usamos el modelo rápido de la familia Gemini
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                
                system_prompt = (
                    "Eres un asistente útil. Usa el siguiente contexto del video para responder la pregunta.\n\n"
                    "Contexto:\n{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(st.session_state.vectorstore.as_retriever(), question_answer_chain)

                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})