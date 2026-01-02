import streamlit as st
from rag import get_conversational_rag_chain

def llm_stream(agent, messages):
    response_message = ""
    
    for token, metadata in agent.stream({"messages": messages}, stream_mode="messages"):
        node = metadata['langgraph_node']
        content_blocks = token.content_blocks
        
        # Procesar cada bloque de contenido
        for block in content_blocks:
            if block.get('type') == 'text' and 'text' in block:
                text_chunk = block['text']
                response_message += text_chunk
                yield text_chunk  # Enviar solo el texto a Streamlit
                
            elif block.get('type') == 'tool_call_chunk':
                # Mostrar cuando se están usando herramientas
                tool_name = block.get('name', 'desconocida')
                if tool_name and tool_name != 'desconocida':
                    st.toast(f"Usando herramienta: {tool_name}")
    
    # Guardar mensaje completo al final
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_message
    })
    
def stream_llm_rag_response(llm_stream, messages):
    """Stream RAG real con fuentes al final."""

    # Limitar historial
    limited_messages = messages[-6:] if len(messages) > 6 else messages

    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = ""
    sources = set()

    # Stream con captura de contexto
    for chunk in conversation_rag_chain.stream(
        {"messages": limited_messages[:-1], "input": limited_messages[-1].content}
    ):
        # Capturar documentos del contexto (llegan primero)
        if "context" in chunk and not sources:
            for doc in chunk["context"]:
                source = doc.metadata.get("source", "")
                page = doc.metadata.get("page_number", "")
                if source:
                    sources.add(
                        f"{source}"
                        + (f" (pag.{page})" if page and page != "None" else "")
                    )

        # Stream de la respuesta
        if "answer" in chunk:
            answer_chunk = chunk["answer"]
            response_message += answer_chunk
            yield answer_chunk

    # Agregar fuentes al final
    if sources:
        sources_text = "\n\n**Fuentes:** " + ", ".join(sorted(sources))
        response_message += sources_text
        yield sources_text

    # Guardar mensaje completo
    st.session_state.messages.append({"role": "assistant", "content": response_message})
