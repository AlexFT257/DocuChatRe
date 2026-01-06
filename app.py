from datetime import datetime
from uuid import uuid4

import streamlit as st
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent import llm_stream, stream_llm_rag_response

# import rag functions
from rag import load_doc_to_db
from tools import calculate, search

st.set_page_config(page_title="DocuChat", page_icon="📄")

st.write("# DocuChat")

st.markdown("""
    DocuChat es uina aplicacion que permite cargar documentos para analizar, comparar y conversar con ellos.
    """)

st.warning(
    "🛑 Se recomienda no subir multiples archivos en cortos periodos de tiempo si estas usando una API key gratuita "
)

st.divider()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

with st.sidebar:
    gemini_api_key = st.text_input(
        "Gemini API Key", key="file_qa_api_key", type="password"
    )
    "[Obten tu API Key](https://aistudio.google.com/app/apikey)"


if not gemini_api_key:
    st.error("Por favor, ingresa tu API Key de Gemini")
else:
    st.session_state.gemini_api_key = gemini_api_key

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1.0,  # Gemini 3.0+ defaults to 1.0
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=gemini_api_key,
    )

    current_date = datetime.now().strftime("%d de %B de %Y")

    agent = create_agent(
        model,
        tools=[search, calculate],
        system_prompt=f"""Eres DocuChat, un asistente inteligente que ayuda a los usuarios.

        INFORMACIÓN TEMPORAL IMPORTANTE:
        - Fecha actual: {current_date}

        Tienes acceso a las siguientes herramientas:
        1. search: Para buscar información actualizada en internet
        2. calculate: Para realizar cálculos matemáticos

        INSTRUCCIONES IMPORTANTES:
        - Cuando uses la herramienta de búsqueda, los resultados son ACTUALES y corresponden a {current_date}
        - Responde de manera clara y útil usando Markdown cuando sea apropiado
        - Si no estás seguro de algo, usa la herramienta de búsqueda para verificar""",
    )

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Sube un documento",
            accept_multiple_files=True,
            type=("txt", "md", "pdf", "docx"),
            on_change=load_doc_to_db(),
            key="rag_docs",
        )

        is_vector_db_loaded = (
            "vector_db" in st.session_state and st.session_state.vector_db is not None
        )
        st.toggle(
            "Use RAG",
            value=is_vector_db_loaded,
            key="use_rag",
            disabled=not is_vector_db_loaded,
        )

        with st.expander(
            f"Documentos en la BD ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"
        ):
            st.write(
                []
                if not is_vector_db_loaded
                else [source for source in st.session_state.rag_sources]
            )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hola, soy DocuChat ¿Que consulta tienes el dia de hoy? Recuerda que puesdes subir documentos en la barra lateral.",
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe aqui tu mensaje"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # changing format to langchain format
            messages = [
                HumanMessage(content=m["content"])
                if m["role"] == "user"
                else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            try:
                if not st.session_state.use_rag:
                    st.write_stream(llm_stream(agent, messages))
                else:
                    st.write_stream(stream_llm_rag_response(model, messages))
            except Exception as e:
                st.error(f"Error: {e}")
