import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import Tuple, Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-Based Localization", ICCV 2017.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self):
        """Registers forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple, index 0 contains gradients w.r.t layer output
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generates a 2D Grad-CAM heatmap for the given input tensor.
        
        Args:
            input_tensor: Shape [1, C, H, W]
            class_idx: Target class index. If None, uses model's predicted class.
            
        Returns:
            Normalized 2D heatmap numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        # Target score for backprop
        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks failed to capture activations or gradients.")

        # Global Average Pooling of gradients across spatial dimensions (H, W) -> alpha_k
        # gradients shape: [1, Channels, H, W]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # [1, Channels, 1, 1]

        # Linear combination of feature maps weighted by alpha_k
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply ReLU to retain features that positively correlate with the target class
        cam = F.relu(cam)

        # Upsample to match original input tensor spatial dimensions
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> Tuple[Image.Image, np.ndarray]:
    """
    Overlays a Grad-CAM heatmap onto a PIL radiograph.
    
    Args:
        original_image: PIL RGB Image
        heatmap: 2D numpy array [0, 1]
        alpha: Blend ratio for heatmap overlay (0 = only image, 1 = only heatmap)
        colormap: OpenCV colormap (default JET)
        
    Returns:
        blended_pil: Blended PIL Image
        colored_heatmap: Heatmap RGB array
    """
    # Resize original image and convert to numpy RGB
    orig_np = np.array(original_image)
    if orig_np.ndim == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)

    h, w = orig_np.shape[:2]

    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Convert to 8-bit unsigned integer [0, 255]
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    blended = np.float32(heatmap_color) * alpha + np.float32(orig_np) * (1.0 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended), heatmap_resized
