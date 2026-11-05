import pytest
import numpy as np
import torch
from PIL import Image

from src.models.architecture import build_model
from src.explainability.gradcam import GradCAM, overlay_heatmap


def test_gradcam_generation():
    """Verify GradCAM generates valid 2D normalized heatmaps."""
    model = build_model(backbone="densenet121", num_classes=2, pretrained=False)
    gradcam = GradCAM(model=model, target_layer=model.target_layer)

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap = gradcam.generate(dummy_input, class_idx=1)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224), f"Expected (224, 224), got {heatmap.shape}"
    assert heatmap.min() >= 0.0 - 1e-6
    assert heatmap.max() <= 1.0 + 1e-6


def test_overlay_heatmap():
    """Verify overlay blending returns valid RGB PIL Image."""
    test_img = Image.new("RGB", (224, 224), color="gray")
    mock_heatmap = np.random.uniform(0.0, 1.0, size=(224, 224))

    blended_pil, resized_map = overlay_heatmap(test_img, mock_heatmap, alpha=0.5)
    assert isinstance(blended_pil, Image.Image)
    assert blended_pil.size == (224, 224)
    assert resized_map.shape == (224, 224)
