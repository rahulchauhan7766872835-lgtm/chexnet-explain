import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class CheXNetModel(nn.Module):
    """
    Deep Neural Network for Chest Radiograph Pathology Detection.
    Implements transfer learning architectures:
      - DenseNet-121 (Standard CheXNet backbone)
      - ResNet-50
      - EfficientNet-B0
    """
    def __init__(
        self,
        backbone: str = "densenet121",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.num_classes = num_classes

        if self.backbone_name == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            self.model = models.densenet121(weights=weights)
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            # Default target layer for Grad-CAM
            self.target_layer = self.model.features.denseblock4.denselayer16.conv2

        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            self.target_layer = self.model.layer4[-1].conv3

        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            self.target_layer = self.model.features[-1][0]

        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose densenet121, resnet50, or efficientnet_b0")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits: [batch_size, num_classes]
        """
        return self.model(x)

    def freeze_feature_extractor(self):
        """
        Freezes feature extractor layers for initial classifier head warm-up.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze classifier
        classifier = getattr(self.model, "classifier", None) or getattr(self.model, "fc", None)
        if classifier:
            for param in classifier.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """
        Unfreezes all model parameters for full end-to-end fine-tuning.
        """
        for param in self.model.parameters():
            param.requires_grad = True


def build_model(
    backbone: str = "densenet121",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_rate: float = 0.2
) -> CheXNetModel:
    """Factory function to build and configure the diagnostic model."""
    return CheXNetModel(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
