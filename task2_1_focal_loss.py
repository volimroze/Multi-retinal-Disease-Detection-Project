import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


# ========================
# Reproducibility
# ========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========================
# Dataset (with labels) for train/val/offsite
# CSV format: first column = image filename, remaining columns = labels (3)
# ========================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row.iloc[0]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row.iloc[1:].values.astype("float32"))  # shape (3,)

        if self.transform:
            img = self.transform(img)

        return img, labels


# ========================
# Dataset (NO labels) for Kaggle onsite submission
# Template CSV contains the correct column names (e.g., id,D,G,A or id,DR,Glaucoma,AMD)
# ========================
class RetinaOnsiteDataset(Dataset):
    def __init__(self, template_csv, image_dir, transform=None):
        self.df = pd.read_csv(template_csv)

        if "id" not in self.df.columns:
            raise ValueError(
                f'Column "id" not found in template CSV. Found: {list(self.df.columns)}. '
                "Use the original onsite_test_submission.csv file."
            )

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "id"]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, img_name


# ========================
# Model builder
# IMPORTANT: loads weights from provided ckpt_resnet18_ep50.pt
# ========================
def build_resnet18(num_classes: int = 3):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ========================
# Task 2.1 Loss: Focal Loss for multi-label classification (logits)
# ========================
class FocalLossMultiLabel(nn.Module):
    """
    Multi-label focal loss for logits.
    - logits: (B, C)
    - targets: (B, C) in {0,1}
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        # BCE per element (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ========================
# Metrics helper (per disease)
# ========================
def compute_and_print_metrics(y_true_np: np.ndarray, y_pred_np: np.ndarray, tag: str):
    disease_names = ["DR", "Glaucoma", "AMD"]
    for i, disease in enumerate(disease_names):
        y_t = y_true_np[:, i]
        y_p = y_pred_np[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro", zero_division=0)
        recall = recall_score(y_t, y_p, average="macro", zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        print(f"\n{disease} Results [{tag}]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}")


# ========================
# Task 2.1: Full fine-tune using Focal Loss
# ========================
def run_task2_1_focal_loss(
    train_csv="train.csv",
    val_csv="val.csv",
    offsite_csv="offsite_test.csv",

    train_image_dir="./images/train",
    val_image_dir="./images/val",
    offsite_image_dir="./images/offsite_test",

    onsite_template_csv="onsite_test_submission.csv",
    onsite_image_dir="./images/onsite_test",

    pretrained_ckpt="./pretrained_backbone/ckpt_resnet18_ep50.pt",

    save_dir="checkpoints",
    best_ckpt_name="task2_1_focal_best_resnet18.pt",

    epochs=10,
    batch_size=16,
    lr=1e-5,
    img_size=256,
    threshold=0.5,

    focal_alpha=0.25,
    focal_gamma=2.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Windows friendly
    num_workers = 0

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Loaders
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform=train_tf)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, transform=eval_tf)
    offsite_ds = RetinaMultiLabelDataset(offsite_csv, offsite_image_dir, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    offsite_loader = DataLoader(offsite_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model + pretrained weights (provided by course resources)
    model = build_resnet18(num_classes=3).to(device)
    state_dict = torch.load(pretrained_ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # Task 2.1 uses full fine-tuning (backbone + head)
    for p in model.parameters():
        p.requires_grad = True

    # Focal loss
    criterion = FocalLossMultiLabel(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train & save best by val loss
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, best_ckpt_name)
    best_val_loss = float("inf")

    print(f"\nTask 2.1: Full fine-tuning with Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    print("Training on train.csv, validating on val.csv")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * imgs.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)

        print(f"[resnet18] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to: {best_path}")

    # Evaluate on offsite test (labels available)
    print("\nEvaluating best Task 2.1 model on OFFSITE test set (labels available)")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for imgs, labels in offsite_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            y_true_list.append(labels.numpy())
            y_pred_list.append(preds)

    y_true_np = np.concatenate(y_true_list, axis=0)
    y_pred_np = np.concatenate(y_pred_list, axis=0)

    compute_and_print_metrics(y_true_np, y_pred_np, tag="resnet18_task2_1_focal")

    # Create Kaggle onsite submission (no labels)
    print("\nGenerating Kaggle ONSITE submission CSV for Task 2.1...")
    onsite_ds = RetinaOnsiteDataset(onsite_template_csv, onsite_image_dir, transform=eval_tf)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_ids, all_preds = [], []
    with torch.no_grad():
        for imgs, img_names in onsite_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_ids.extend(list(img_names))
            all_preds.extend(list(preds))

    template_df = pd.read_csv(onsite_template_csv)
    col_names = template_df.columns.tolist()

    if col_names[0] != "id":
        raise ValueError(f"Template first column must be 'id'. Found: {col_names[0]}")

    disease_cols = col_names[1:]
    if len(disease_cols) != 3:
        raise ValueError(f"Expected 3 disease columns in template. Found: {disease_cols}")

    pred_df = pd.DataFrame(all_preds, columns=disease_cols)
    out_df = pd.DataFrame({"id": all_ids})
    out_df = pd.concat([out_df, pred_df], axis=1)

    # checks
    if out_df.columns.tolist() != col_names:
        raise ValueError(f"Column mismatch. Expected: {col_names}, got: {out_df.columns.tolist()}")
    if len(out_df) != len(template_df):
        raise ValueError(f"Row mismatch. Expected: {len(template_df)}, got: {len(out_df)}")

    out_csv = "task2_1_onsite_submission.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(out_df.head())


if __name__ == "__main__":
    set_seed(42)
    run_task2_1_focal_loss(
        epochs=10,
        batch_size=16,
        lr=1e-5,
        img_size=256,
        threshold=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0
    )
