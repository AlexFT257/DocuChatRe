import os
import shutil
from time import time

import streamlit as st
from langchain_chroma.vectorstores import Chroma
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.llms import GoogleGenerativeAI
from unstructured.chunking.basic import chunk_elements
from unstructured.cleaners.core import clean, replace_unicode_quotes
from unstructured.partition.auto import partition

MAX_HISTORY_MESSAGES = 10
RETRIEVER_K = 5
RELEVANCE_THRESHOLD = 0.7


def load_doc_to_db():
    # Comprobar si se han subido documentos
    if "rag_docs" not in st.session_state or not st.session_state.rag_docs:
        st.error("No se subio ningun documento")
        return

    docs = []

    if len(st.session_state.rag_sources) >= 10:
        st.error("Solo se pueden cargar hasta 10 documentos, elimine alguno")
        return

    # Crear carpeta de sesión una sola vez
    session_folder = f"source_files_{st.session_state['session_id']}"
    os.makedirs(session_folder, exist_ok=True)

    for doc_file in st.session_state.rag_docs:
        # Verificar si el documento ya ha sido cargado
        if doc_file.name in st.session_state.rag_sources:
            # st.toast("Documento ya cargado")
            continue

        # hacer el archivo
        file_path = f"./{session_folder}/{doc_file.name}"
        with open(file_path, "wb") as file:
            file.write(doc_file.read())

        try:
            # Chunking (unstructured necesita el file path en el nombre)
            with st.spinner(f"Procesando {doc_file.name}..."):
                elements = partition(filename=file_path)

            for element in elements:
                element.text = clean(element.text, extra_whitespace=True)
                element.text = replace_unicode_quotes(element.text)

            chunks = chunk_elements(elements)

            # Transformar chunks de unstructured en documentos
            for i, chunk in enumerate(chunks):
                metadata_dict = dict(chunk.metadata.to_dict())

                clean_metadata = {
                    "source": doc_file.name,
                    "chunk_id": i,
                    "page_number": metadata_dict.get("page_number", "None"),
                    "filetype": metadata_dict.get("filetype", "None"),
                    "filename": metadata_dict.get("filename", "None"),
                    # Convertir listas a strings si existen
                    "languages": ", ".join(metadata_dict.get("languages", []))
                    if metadata_dict.get("languages")
                    else None,
                }

                doc = Document(page_content=chunk.text, metadata=clean_metadata)
                docs.append(doc)

            st.session_state.rag_sources.append(doc_file.name)

        except Exception as e:
            st.toast(
                f"Error al procesar el documento {doc_file.name}: {str(e)}", icon="🚨"
            )

    if os.path.exists(session_folder):
        shutil.rmtree(session_folder)

    if docs:
        add_docs(docs)
        st.toast(
            f"Documento {str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]} cargado",
            icon="✅",
        )


def initialize_vector_db(docs):
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(
            api_key=st.session_state.gemini_api_key,
            model="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
        ),
        # para aislar los documentos por sesión/usuario
        collection_name=f"{int(time() * 1000)}_" + st.session_state["session_id"],
    )

    return vector_db


def add_docs(docs):
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
        return

    st.session_state.vector_db.add_documents(docs)


# RAG
def get_conversational_rag_chain(agent):
    retriever = st.session_state.vector_db.as_retriever(
        search_kwargs={"k": RETRIEVER_K}
    )

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            (
                "user",
                "Genera una consulta de búsqueda específica basada en lo anterior.",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(agent, retriever, retriever_prompt)

    main_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Eres DocuChat, un asistente que analiza documentos cargados por el usuario.

    Responde usando principalmente el contexto. Si la información no está en el contexto, dilo claramente.
    Menciona siempre de qué documento viene la información.
    Responde usando Markdown.

    Contexto:
    {context}""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(agent, main_prompt)
    return create_retrieval_chain(retriever_chain, qa_chain)
