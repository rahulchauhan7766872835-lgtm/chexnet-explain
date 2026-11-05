import base64
import io
import time
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch
import yaml
import numpy as np

from src.models.architecture import build_model
from src.data.dataset import get_transforms
from src.explainability.gradcam import GradCAM, overlay_heatmap


# Pydantic Response Schema
class DiagnosticResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    gradcam_overlay_base64: Optional[str] = None
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    model_backbone: str
    num_classes: int
    device: str
    version: str = "1.0.0"


# Initialize FastAPI App
app = FastAPI(
    title="CheXNet-Explain Medical Diagnostic API",
    description="Deep Learning REST API for Chest Radiograph Pathology Detection with Grad-CAM Explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL_STATE = {
    "model": None,
    "gradcam": None,
    "transforms": None,
    "classes": ["NORMAL", "PNEUMONIA"],
    "device": torch.device("cpu"),
    "backbone": "densenet121",
    "image_size": 224
}


def load_application_model():
    """Loads configuration and initializes model and Grad-CAM hooks."""
    config_path = Path("configs/config.yaml")
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    classes = config.get("model", {}).get("classes", ["NORMAL", "PNEUMONIA"])
    backbone = config.get("model", {}).get("backbone", "densenet121")
    image_size = config.get("data", {}).get("image_size", 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        backbone=backbone,
        num_classes=len(classes),
        pretrained=True
    ).to(device)

    # Load trained weights if checkpoint exists
    checkpoint_path = Path(config.get("paths", {}).get("best_model_path", "checkpoints/best_densenet121.pth"))
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"[API] Loaded trained checkpoint from {checkpoint_path}")
    else:
        print("[API] Running with ImageNet pretrained transfer learning weights (demo mode)")

    model.eval()
    gradcam = GradCAM(model=model, target_layer=model.target_layer)
    transforms = get_transforms(image_size=image_size, is_train=False)

    MODEL_STATE.update({
        "model": model,
        "gradcam": gradcam,
        "transforms": transforms,
        "classes": classes,
        "device": device,
        "backbone": backbone,
        "image_size": image_size
    })


@app.on_event("startup")
def startup_event():
    load_application_model()


@app.get("/")
def root():
    """Root redirect to interactive Swagger documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_backbone=MODEL_STATE["backbone"],
        num_classes=len(MODEL_STATE["classes"]),
        device=str(MODEL_STATE["device"])
    )


@app.post("/predict", response_model=DiagnosticResponse)
async def predict_radiograph(file: UploadFile = File(...), return_heatmap: bool = True):
    """
    Accepts an uploaded chest radiograph image, executes inference, and generates
    a Grad-CAM explainability heatmap overlay.
    """
    if MODEL_STATE["model"] is None:
        load_application_model()

    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    start_time = time.time()

    # Preprocess
    transforms = MODEL_STATE["transforms"]
    device = MODEL_STATE["device"]
    input_tensor = transforms(pil_image).unsqueeze(0).to(device)

    # Inference
    model = MODEL_STATE["model"]
    classes = MODEL_STATE["classes"]

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    # Explainability: Grad-CAM
    overlay_base64 = None
    if return_heatmap and MODEL_STATE["gradcam"] is not None:
        try:
            heatmap = MODEL_STATE["gradcam"].generate(input_tensor, class_idx=pred_idx)
            overlay_pil, _ = overlay_heatmap(pil_image, heatmap, alpha=0.45)
            
            buf = io.BytesIO()
            overlay_pil.save(buf, format="PNG")
            overlay_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as cam_err:
            print(f"[WARNING] Grad-CAM generation error: {cam_err}")

    elapsed_ms = round((time.time() - start_time) * 1000, 2)
    prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}

    return DiagnosticResponse(
        predicted_class=pred_class,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
        inference_time_ms=elapsed_ms,
        gradcam_overlay_base64=overlay_base64
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
