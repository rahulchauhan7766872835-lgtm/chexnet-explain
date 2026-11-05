import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    """Verify API health status endpoint returns 200 and schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_backbone" in data


def test_api_predict_endpoint():
    """Verify /predict endpoint handles image upload and returns valid response."""
    # Create small test in-memory image
    img = Image.new("RGB", (224, 224), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict?return_heatmap=true",
        files={"file": ("test.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ["NORMAL", "PNEUMONIA"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert "inference_time_ms" in data
    assert data["gradcam_overlay_base64"] is not None
