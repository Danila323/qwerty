import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    """Тест главной страницы"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hugging Face Text Classification API"}

def test_classify_text():
    test_data = {"text": "I love machine learning"}
    response = client.post("/classify", json=test_data)
    
    assert response.status_code == 200
    response_data = response.json()
    assert "Текст" in response_data
    assert "Лейбл" in response_data
    assert "Точность" in response_data
    assert isinstance(response_data["Точность"], float)

def test_invalid_input():
    """Тест на некорректные входные данные"""
    response = client.post("/classify", json={"wrong_key": "test"})
    assert response.status_code == 422

@pytest.mark.parametrize("text", [
    "Positive text",
    "Negative text",
    "",
    "Very long text..."*30
])
def test_classify_with_different_inputs(text):
    """Параметризованный тест с разными входными данными"""
    response = client.post("/classify", json={"text": text})
    assert response.status_code == 200