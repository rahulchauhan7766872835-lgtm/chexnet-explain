import argparse
import os
from pathlib import Path
import time
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

from src.models.architecture import build_model
from src.data.dataset import (
    ChestXRayDataset,
    get_transforms,
    scan_dataset,
    compute_class_weights,
    generate_mock_radiograph,
)
from src.training.metrics import calculate_clinical_metrics


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / (total if total > 0 else 1)
    epoch_acc = correct / (total if total > 0 else 1)
    return epoch_loss, epoch_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """Evaluates the model on validation/test set and returns loss and metrics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of pneumonia (class 1)
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            total += labels.size(0)

    val_loss = running_loss / (total if total > 0 else 1)
    metrics = calculate_clinical_metrics(all_labels, all_preds, all_probs)
    return val_loss, metrics


def prepare_sample_dataset_if_needed(sample_dir: Path):
    """Creates a sample dataset if no real dataset is present."""
    if not (sample_dir / "NORMAL").exists() or len(list((sample_dir / "NORMAL").glob("*.png"))) < 5:
        print("[INFO] Creating mock radiograph dataset for smoke test/demo...")
        for i in range(12):
            generate_mock_radiograph(sample_dir / "NORMAL" / f"normal_{i:03d}.png", is_pneumonia=False)
            generate_mock_radiograph(sample_dir / "PNEUMONIA" / f"pneumonia_{i:03d}.png", is_pneumonia=True)
        print("[INFO] Mock dataset generated at:", sample_dir)


def train_pipeline(config_path: str = "configs/config.yaml"):
    """Full training pipeline with early stopping and checkpointing."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Determine compute device
    device_str = config["training"].get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[INFO] Using compute device: {device}")

    # Dataset discovery
    data_dir = Path(config["paths"]["data_dir"])
    sample_dir = Path(config["paths"]["sample_dir"])
    target_data_dir = data_dir if data_dir.exists() and any(data_dir.iterdir()) else sample_dir

    if not target_data_dir.exists() or not any(target_data_dir.iterdir()):
        prepare_sample_dataset_if_needed(target_data_dir)

    classes = config["model"]["classes"]
    file_paths, labels = scan_dataset(str(target_data_dir), classes=classes)

    if len(file_paths) == 0:
        prepare_sample_dataset_if_needed(target_data_dir)
        file_paths, labels = scan_dataset(str(target_data_dir), classes=classes)

    print(f"[INFO] Discovered {len(file_paths)} radiographs across classes: {classes}")

    # Train / Val Split (stratified)
    indices = np.arange(len(file_paths))
    np.random.seed(config["training"]["seed"])
    np.random.shuffle(indices)

    val_count = max(4, int(len(indices) * config["data"]["val_split"]))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_files = [file_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_files = [file_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    train_dataset = ChestXRayDataset(train_files, train_labels, transform=get_transforms(config["data"]["image_size"], is_train=True))
    val_dataset = ChestXRayDataset(val_files, val_labels, transform=get_transforms(config["data"]["image_size"], is_train=False))

    batch_size = config["data"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Class-weighted CrossEntropy
    class_weights = compute_class_weights(train_labels, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Build model
    model = build_model(
        backbone=config["model"]["backbone"],
        num_classes=len(classes),
        pretrained=config["model"]["pretrained"],
        dropout_rate=config["model"]["dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config["training"]["scheduler_patience"], factor=0.5
    )

    best_f1 = 0.0
    patience_counter = 0
    patience_limit = config["training"]["early_stopping_patience"]
    epochs = config["training"]["epochs"]

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = Path(config["paths"]["best_model_path"])

    print(f"[INFO] Commencing training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        val_f1 = val_metrics["f1_score"]
        scheduler.step(val_f1)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} Sens: {val_metrics['sensitivity']:.4f} Spec: {val_metrics['specificity']:.4f} AUROC: {val_metrics['auroc']:.4f}"
        )

        # Checkpoint if best F1
        if val_f1 >= best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": val_metrics,
                "config": config
            }, best_model_path)
            print(f"  >>> Best model saved with F1: {best_f1:.4f} to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"[INFO] Early stopping triggered after {epoch} epochs.")
                break

    print(f"[SUCCESS] Training complete. Best Validation F1: {best_f1:.4f}")
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CheXNet Radiograph Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    train_pipeline(args.config)
