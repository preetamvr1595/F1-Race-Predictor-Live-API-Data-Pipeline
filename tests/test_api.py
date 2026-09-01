"""
Unit Tests for Jolpica-F1 API Client.
"""

from src.api_client import JolpicaF1Client

def test_client_init():
    client = JolpicaF1Client()
    assert client.base_url == "http://api.jolpi.ca/ergast/f1"
    assert client.timeout == 10
