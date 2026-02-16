import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Chat Pro", page_icon="🧠")
st.title("🧠 Gemini 1.5 Pro (Chatbot)")
st.caption("Usando el modelo más potente y estable de Google.")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("🔑 Llave de Acceso")
    # Pide la clave aquí para no guardarla en código
    api_key = st.text_input("Pega tu Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    if st.button("🗑️ Limpiar Conversación"):
        st.session_state.chat_history = []
        st.rerun()

# --- VALIDACIÓN ---
if not api_key:
    st.warning("👈 Necesito tu API Key para activar el cerebro del modelo.")
    st.stop()

# --- MEMORIA DEL CHAT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- MOSTRAR MENSAJES ANTERIORES ---
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# --- INTERACCIÓN ---
user_input = st.chat_input("Escribe algo complicado...")

if user_input:
    # 1. Guardar y mostrar lo que tú escribes
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Respuesta de la IA
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # AQUÍ ESTÁ EL CAMBIO: Usamos el modelo PRO
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=0.7
            )
            
            with st.spinner("Pensando a fondo..."):
                response = llm.invoke(st.session_state.chat_history)
                message_placeholder.markdown(response.content)
                
            # Guardamos la respuesta en memoria
            st.session_state.chat_history.append(AIMessage(content=response.content))
            
        except Exception as e:
            st.error(f"Error: {e}")