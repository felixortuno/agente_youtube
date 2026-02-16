import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Chat Gemini Pro", page_icon="🤖")
st.title("🤖 Chat Gemini Pro")
st.caption("Usando el modelo estándar (gemini-pro) para máxima compatibilidad.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("🔑 Llave de Acceso")
    api_key = st.text_input("Pega tu Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    if st.button("🗑️ Borrar Historial"):
        st.session_state.chat_history = []
        st.rerun()

# --- VALIDACIÓN ---
if not api_key:
    st.warning("👈 Pega tu API Key para empezar.")
    st.stop()

# --- MEMORIA DEL CHAT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- MOSTRAR MENSAJES ---
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- INTERACCIÓN ---
user_input = st.chat_input("Escribe algo...")

if user_input:
    # 1. Tu mensaje
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Respuesta de la IA
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # --- CAMBIO CLAVE: USAMOS 'gemini-pro' ---
            # Este modelo es el más compatible a nivel mundial
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.7
            )
            
            with st.spinner("Pensando..."):
                response = llm.invoke(st.session_state.chat_history)
                message_placeholder.markdown(response.content)
                
            # Guardar respuesta
            st.session_state.chat_history.append(AIMessage(content=response.content))
            
        except Exception as e:
            st.error(f"Error de conexión: {e}")
            st.info("💡 Si sigue fallando, crea una API Key nueva en aistudio.google.com y asegúrate de elegir un proyecto 'Free'.")