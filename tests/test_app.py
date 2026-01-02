# from unittest.mock import MagicMock, Mock, patch

import pytest
from streamlit.testing.v1 import AppTest


def test_app_render_without_key():
    at = AppTest.from_file("../app.py", default_timeout=10).run()

    assert "Por favor, ingresa tu API Key de Gemini" in at.error[0].value
    # assert len(at.error) == 1


def test_app_accepts_api_key():
    at = AppTest.from_file("../app.py")
    at.run(timeout=10)

    at.text_input("file_qa_api_key").input("dummy-key").run()
    assert not at.exception

