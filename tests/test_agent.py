import pytest
from unittest.mock import MagicMock, Mock, patch
from agent import llm_stream, stream_llm_rag_response

@pytest.fixture
def mock_streamlit(monkeypatch):
    mock_st = Mock()
    mock_st.toast = Mock()
    mock_st.session_state.messages = []
    monkeypatch.setattr("agent.st", mock_st)

    return mock_st

@pytest.fixture
def mock_streamlit_session():
    with patch("agent.st") as mock_st:
        mock_st.session_state.gemini_api_key = "test_api_key"
        mock_st.session_state.messages = []
        yield mock_st


def test_llm_stream(mock_streamlit):
    mock_agent = Mock()

    mock_agent.stream.return_value = [
        (
            Mock(content_blocks=[{"type": "text", "text": "Hola "}]),
            {"langgraph_node": "test_node"},
        ),
        (
            Mock(content_blocks=[{"type": "text", "text": "mundo"}]),
            {"langgraph_node": "test_node"},
        ),
    ]

    messages = [{"role": "user", "content": "test"}]

    result = list(llm_stream(mock_agent, messages))

    assert result == ["Hola ", "mundo"]
    assert len(mock_streamlit.session_state.messages) == 1
    assert mock_streamlit.session_state.messages[0]["content"] == "Hola mundo"

def test_stream_llm_rag_response(mock_streamlit_session):
    mock_chain = Mock()
    mock_chain.stream.return_value = [
        {
            "context":
                [Mock(metadata={"source": "doc1.pdf", "page_number": 1})
            ]
        },
        {"answer": "Esta es "},
        {"answer": "la respuesta"}
    ]
    
    with patch("agent.get_conversational_rag_chain", return_value=mock_chain):
        messages = [Mock(content="test query")]
        
        result = list(stream_llm_rag_response(Mock(), messages))

        assert "Esta es " in result
        assert "la respuesta" in result
        assert any("Fuentes" in str(chunk) for chunk in result)



def test_llm_stream_tool_toast(mock_streamlit):
    mock_agent = Mock()

    mock_agent.stream.return_value = [
        (
            Mock(
                content_blocks=[{"type": "tool_call_chunk", "name": "buscar_documento"}]
            ),
            {"langgraph_node": "test_node"},
        )
    ]

    messages = [{"role": "user", "content": "test"}]
    list(llm_stream(mock_agent, messages))
    # Verificar que se llamó toast
    mock_streamlit.toast.assert_called_once_with("Usando herramienta: buscar_documento")
