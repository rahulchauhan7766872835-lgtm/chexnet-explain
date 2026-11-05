import pytest
import torch
from src.models.architecture import build_model, CheXNetModel


def test_densenet121_initialization():
    """Verify DenseNet-121 builds with correct output classes and target layer."""
    model = build_model(backbone="densenet121", num_classes=2, pretrained=False)
    assert isinstance(model, CheXNetModel)
    assert model.target_layer is not None


def test_model_forward_pass_shape():
    """Verify forward pass returns expected tensor shapes."""
    model = build_model(backbone="densenet121", num_classes=2, pretrained=False)
    model.eval()

    # Batch of 2 RGB images at 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"


def test_resnet50_backbone():
    """Verify ResNet-50 backbone support."""
    model = build_model(backbone="resnet50", num_classes=2, pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 2)
