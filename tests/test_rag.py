from unittest import mock
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
from langchain_core.documents import Document

from rag import add_docs, initialize_vector_db, load_doc_to_db


class MockSessionState(dict):
    """Mock personalizado para session_state que se comporta como dict y objeto"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )


@pytest.fixture
def mock_streamlit():
    with patch("rag.st") as mock_st:
        # Usar MockSessionState personalizado, se comporta raro si no
        session_state = MockSessionState(
            {
                "gemini_api_key": "test_api_key",
                "messages": [],
                "session_id": "test_session_id",
                "rag_sources": [],
                "rag_docs": [],
            }
        )

        mock_st.session_state = session_state

        # Configurar métodos
        mock_st.error = Mock()
        mock_st.toast = Mock()
        mock_st.spinner = MagicMock()
        mock_st.spinner.return_value.__enter__ = Mock(return_value=None)
        mock_st.spinner.return_value.__exit__ = Mock(return_value=None)

        yield mock_st


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Contenido de prueba 1",
            metadata={"source": "doc1.pdf", "page_number": 1},
        ),
        Document(
            page_content="Contenido de prueba 2",
            metadata={"source": "doc2.pdf", "page_number": 2},
        ),
    ]


@pytest.fixture
def mock_uploaded_file():
    mock_file = Mock()
    mock_file.name = "test_document.pdf"
    mock_file.read.return_value = b"Contenido del archivo"
    return mock_file


@patch("rag.Chroma")
@patch("rag.GoogleGenerativeAIEmbeddings")
def test_initialize_vector_db(
    mock_embeddings, mock_chroma, mock_streamlit, sample_docs
):
    result = initialize_vector_db(sample_docs)
    mock_chroma.from_documents.assert_called_once()
    mock_embeddings.assert_called_once()


def test_add_docs_initialization(mock_streamlit, sample_docs):
    # Eliminar vector_db si existe
    if "vector_db" in mock_streamlit.session_state:
        del mock_streamlit.session_state["vector_db"]

    with patch("rag.initialize_vector_db") as mock_init:
        mock_init.return_value = Mock()
        add_docs(sample_docs)
        mock_init.assert_called_once_with(sample_docs)


def test_add_docs_with_existing_db(mock_streamlit, sample_docs):
    mock_vector_db = Mock()
    mock_streamlit.session_state.vector_db = mock_vector_db

    add_docs(sample_docs)

    mock_vector_db.add_documents.assert_called_once_with(sample_docs)


def test_load_doc_to_db_without_file(mock_streamlit):
    mock_streamlit.session_state.rag_docs = []
    load_doc_to_db()
    mock_streamlit.error.assert_called_once()


def test_load_doc_to_db_docs_limit(mock_streamlit, mock_uploaded_file):
    mock_streamlit.session_state.rag_sources = [f"doc{i}" for i in range(10)]
    mock_streamlit.session_state.rag_docs = [mock_uploaded_file]
    load_doc_to_db()
    mock_streamlit.error.assert_called_once()


def test_load_doc_to_db_duplicated_docs(mock_streamlit, mock_uploaded_file):
    mock_streamlit.session_state.rag_sources = [mock_uploaded_file.name]
    mock_streamlit.session_state.rag_docs = [mock_uploaded_file]

    with patch("rag.add_docs") as mock_add:
        with patch("rag.os.path.exists", return_value=False):
            with patch("rag.os.makedirs"):
                load_doc_to_db()
                mock_add.assert_not_called()


@patch("rag.shutil.rmtree")
@patch("rag.os.path.exists")
@patch("rag.os.makedirs")
@patch("rag.add_docs")
@patch("rag.chunk_elements")
@patch("rag.partition")
@patch("rag.clean")
@patch("rag.replace_unicode_quotes")
@patch("builtins.open", new_callable=mock_open)
def test_load_doc_to_db_success(
    mock_file_open,
    mock_replace_unicode,
    mock_clean,
    mock_partition,
    mock_chunk_elements,
    mock_add_docs,
    mock_makedirs,
    mock_exists,
    mock_rmtree,
    mock_streamlit,
    mock_uploaded_file,
):
    mock_streamlit.session_state.rag_docs = [mock_uploaded_file]
    mock_streamlit.session_state.rag_sources = []

    mock_element = Mock()
    mock_element.text = "Texto de prueba"
    mock_element.metadata.to_dict.return_value = {
        "page_number": 1,
        "filetype": "application/pdf",
        "filename": "test_document.pdf",
        "languages": ["es"],
    }

    mock_chunk = Mock()
    mock_chunk.text = "Texto de prueba procesado"
    mock_chunk.metadata = mock_element.metadata

    mock_partition.return_value = [mock_element]
    mock_chunk_elements.return_value = [mock_chunk]
    mock_exists.return_value = True
    mock_clean.return_value = "Texto limpio"
    mock_replace_unicode.return_value = "Texto sin unicode"

    load_doc_to_db()

    mock_partition.assert_called_once()
    mock_chunk_elements.assert_called_once()
    mock_add_docs.assert_called_once()

    call_args = mock_add_docs.call_args[0][0]
    assert len(call_args) > 0
    assert all(isinstance(doc, Document) for doc in call_args)

    mock_rmtree.assert_called_once()
    mock_streamlit.toast.assert_called_once()

    assert mock_uploaded_file.name in mock_streamlit.session_state.rag_sources


@patch("rag.shutil.rmtree")
@patch("rag.os.path.exists")
@patch("rag.os.makedirs")
@patch("rag.partition")
@patch("builtins.open", new_callable=mock_open)
def test_load_doc_to_db_processing_error(
    mock_file_open,
    mock_partition,
    mock_makedirs,
    mock_exists,
    mock_rmtree,
    mock_streamlit,
    mock_uploaded_file,
):
    mock_streamlit.session_state.rag_docs = [mock_uploaded_file]
    mock_streamlit.session_state.rag_sources = []

    mock_partition.side_effect = Exception("Error de procesamiento")
    mock_exists.return_value = True

    load_doc_to_db()

    mock_streamlit.toast.assert_called()
    call_args = mock_streamlit.toast.call_args[0][0]
    assert "Error al procesar el documento" in call_args
    assert mock_uploaded_file.name in call_args

    mock_rmtree.assert_called_once()

    assert mock_uploaded_file.name not in mock_streamlit.session_state.rag_sources


@patch("rag.shutil.rmtree")
@patch("rag.os.path.exists")
@patch("rag.os.makedirs")
@patch("rag.add_docs")
@patch("rag.chunk_elements")
@patch("rag.partition")
@patch("rag.clean")
@patch("rag.replace_unicode_quotes")
@patch("builtins.open", new_callable=mock_open)
def test_load_doc_to_db_multiple_files(
    mock_file_open,
    mock_replace_unicode,
    mock_clean,
    mock_partition,
    mock_chunk_elements,
    mock_add_docs,
    mock_makedirs,
    mock_exists,
    mock_rmtree,
    mock_streamlit,
):
    mock_file1 = Mock()
    mock_file1.name = "doc1.pdf"
    mock_file1.read.return_value = b"Contenido 1"

    mock_file2 = Mock()
    mock_file2.name = "doc2.pdf"
    mock_file2.read.return_value = b"Contenido 2"

    mock_streamlit.session_state.rag_docs = [mock_file1, mock_file2]
    mock_streamlit.session_state.rag_sources = []

    mock_element = Mock()
    mock_element.text = "Texto"
    mock_element.metadata.to_dict.return_value = {
        "page_number": 1,
        "filetype": "application/pdf",
        "filename": "test.pdf",
        "languages": ["es"],
    }

    mock_chunk = Mock()
    mock_chunk.text = "Texto procesado"
    mock_chunk.metadata = mock_element.metadata

    mock_partition.return_value = [mock_element]
    mock_chunk_elements.return_value = [mock_chunk]
    mock_exists.return_value = True
    mock_clean.return_value = "Texto limpio"
    mock_replace_unicode.return_value = "Texto sin unicode"

    load_doc_to_db()

    assert mock_partition.call_count == 2
    assert len(mock_streamlit.session_state.rag_sources) == 2
    assert "doc1.pdf" in mock_streamlit.session_state.rag_sources
    assert "doc2.pdf" in mock_streamlit.session_state.rag_sources


def test_load_doc_to_db_empty_after_filtering(mock_streamlit, mock_uploaded_file):
    mock_streamlit.session_state.rag_sources = [mock_uploaded_file.name]
    mock_streamlit.session_state.rag_docs = [mock_uploaded_file]

    with patch("rag.add_docs") as mock_add:
        with patch("rag.os.path.exists", return_value=False):
            with patch("rag.os.makedirs"):
                load_doc_to_db()

                mock_add.assert_not_called()
                mock_streamlit.toast.assert_not_called()
