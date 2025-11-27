import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.api_pytest import app

client = TestClient(app)


@pytest.fixture
def mock_model():
    # Patch the model object inside the API
    with patch("src.api.api_pytest.model", new=MagicMock()) as mock_model:

        # Create fake pipeline with predict()
        mock_pipeline = MagicMock()

        # Attach pipeline to model
        mock_model.pipeline = mock_pipeline

        yield mock_model


def test_predict(mock_model):
    response = client.post(
        "/predict",
        json={"text": "I love this!"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data

    # Ensure pipeline.predict was called
    mock_model.pipeline.predict.assert_called_once()


def test_missing_text_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_invalid_json():
    response = client.post("/predict", data="INVALID")
    assert response.status_code == 422
