import io
from pathlib import Path
from typing import List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ("NORMAL", "PNEUMONIA"),
    title: str = "Diagnostic Confusion Matrix"
) -> Image.Image:
    """
    Renders a seaborn confusion matrix with counts and percentages.
    Returns a PIL Image.
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Pathology", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def plot_triage_dashboard(
    original_img: Image.Image,
    heatmap: np.ndarray,
    overlay_img: Image.Image,
    diagnosis: str,
    confidence: float,
    metrics: dict = None
) -> Image.Image:
    """
    Creates a publication-quality triage dashboard:
    Original Radiograph | Grad-CAM Heatmap | Blended Diagnostic Overlay
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # 1. Original
    axes[0].imshow(original_img)
    axes[0].set_title("Input Chest Radiograph", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # 2. Grad-CAM Heatmap
    im1 = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Class Activation Map", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Overlay
    axes[2].imshow(overlay_img)
    axes[2].set_title(f"Clinical Overlay: {diagnosis} ({confidence*100:.1f}%)", fontsize=12, fontweight="bold", color="darkred" if diagnosis == "PNEUMONIA" else "navy")
    axes[2].axis("off")

    plt.suptitle("CheXNet-Explain: AI Radiograph Pathology Triage", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)
