# task2_2_class_balanced_full_finetune.py
# Task 2.2: Class-Balanced Loss (re-weight BCE according to class frequency)
# Full fine-tuning: backbone + classifier are trainable
#
# Expected CSV header (as in your screenshot): id,D,G,A

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
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
# CSV format: id,D,G,A
# ========================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform=None):
        self.data = pd.read_csv(csv_file)

        # Expected header based on your screenshot
        expected_cols = ["id", "D", "G", "A"]
        for c in expected_cols:
            if c not in self.data.columns:
                raise ValueError(
                    f"Missing column '{c}' in {csv_file}. Found: {list(self.data.columns)}. "
                    f"Expected at least: {expected_cols}"
                )

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_name = row["id"]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[["D", "G", "A"]].values.astype("float32"))  # shape (3,)

        if self.transform:
            img = self.transform(img)

        return img, labels


# ========================
# Dataset (NO labels) for Kaggle onsite submission
# Template CSV format: id,D,G,A (as your project uses)
# ========================
class RetinaOnsiteDataset(Dataset):
    def __init__(self, template_csv: str, image_dir: str, transform=None):
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

    def __getitem__(self, idx: int):
        img_name = self.df.loc[idx, "id"]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, img_name


# ========================
# Model builder
# ========================
def build_resnet18(num_classes: int = 3):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ========================
# Metrics helper (per label)
# ========================
def compute_and_print_metrics(y_true_np: np.ndarray, y_pred_np: np.ndarray, tag: str):
    disease_names = ["DR(D)", "Glaucoma(G)", "AMD(A)"]

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
# Compute class-balanced pos_weight for BCEWithLogitsLoss
# For multi-label BCE, a common class-balanced approach is:
# pos_weight[c] = (N - pos_c) / pos_c
# ========================
def compute_pos_weight_from_train_csv(train_csv: str) -> torch.Tensor:
    df = pd.read_csv(train_csv)

    # based on your screenshot header: id,D,G,A
    for col in ["D", "G", "A"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {train_csv}. Found: {list(df.columns)}")

    N = len(df)
    pos = df[["D", "G", "A"]].sum(axis=0).astype(float).values  # shape (3,)

    # Avoid division by zero (shouldn't happen, but safe)
    eps = 1e-6
    pos = np.maximum(pos, eps)

    neg = N - pos
    pos_weight = neg / pos  # shape (3,)

    return torch.tensor(pos_weight, dtype=torch.float32)


# ========================
# Task 2.2: Full fine-tuning with Class-Balanced BCE
# ========================
def run_task2_2_class_balanced(
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
    best_ckpt_name="task2_2_cb_best_resnet18.pt",
    out_csv_name="task2_2_onsite_submission.csv",
    epochs=10,
    batch_size=16,
    lr=1e-5,
    img_size=256,
    threshold=0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Windows-friendly
    num_workers = 0

    # Transforms (same pattern you used in earlier tasks)
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

    # Data
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform=train_tf)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, transform=eval_tf)
    offsite_ds = RetinaMultiLabelDataset(offsite_csv, offsite_image_dir, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    offsite_loader = DataLoader(offsite_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model + load provided pretrained backbone checkpoint
    model = build_resnet18(num_classes=3).to(device)

    state_dict = torch.load(pretrained_ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # Task 2.2 uses full fine-tuning (backbone + head)
    for p in model.parameters():
        p.requires_grad = True

    # Class-balanced weights for BCEWithLogitsLoss
    pos_weight = compute_pos_weight_from_train_csv(train_csv).to(device)
    print(f"Task 2.2: Using Class-Balanced BCE (pos_weight) = {pos_weight.detach().cpu().numpy()}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, best_ckpt_name)
    best_val_loss = float("inf")

    print("\nTask 2.2: Full fine-tuning with Class-Balanced Loss (train.csv -> val.csv)")
    for epoch in range(epochs):
        # ---- train ----
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

        # ---- val ----
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

    # ========================
    # Evaluate on OFFSITE test (has labels)
    # ========================
    print("\nEvaluating best Task 2.2 model on OFFSITE test set (labels available)")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    y_true_list = []
    y_pred_list = []

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

    compute_and_print_metrics(y_true_np, y_pred_np, tag="resnet18_task2_2_class_balanced")

    # ========================
    # Create Kaggle ONSITE submission (NO labels)
    # ========================
    print("\nGenerating Kaggle ONSITE submission CSV for Task 2.2...")
    template_df = pd.read_csv(onsite_template_csv)
    col_names = template_df.columns.tolist()

    if len(col_names) != 4 or col_names[0] != "id":
        raise ValueError(
            f"Unexpected onsite template header: {col_names}. "
            f"Expected something like ['id','D','G','A']."
        )

    disease_cols = col_names[1:]  # typically ['D','G','A']

    onsite_ds = RetinaOnsiteDataset(onsite_template_csv, onsite_image_dir, transform=eval_tf)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for imgs, img_names in onsite_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_ids.extend(list(img_names))
            all_preds.extend(list(preds))

    pred_df = pd.DataFrame(all_preds, columns=disease_cols)
    out_df = pd.DataFrame({"id": all_ids})
    out_df = pd.concat([out_df, pred_df], axis=1)

    # Final checks
    if out_df.columns.tolist() != col_names:
        raise ValueError(f"Column mismatch. Expected: {col_names}, Got: {out_df.columns.tolist()}")
    if len(out_df) != len(template_df):
        raise ValueError(f"Row count mismatch. Expected: {len(template_df)}, Got: {len(out_df)}")

    out_df.to_csv(out_csv_name, index=False)
    print(f"Saved: {out_csv_name}")
    print(out_df.head())


if __name__ == "__main__":
    set_seed(42)
    run_task2_2_class_balanced(
        epochs=10,
        batch_size=16,
        lr=1e-5,
        img_size=256,
        threshold=0.5,
    )
