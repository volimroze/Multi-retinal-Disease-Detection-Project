import os
import random
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet18

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
# Dataset: labeled (train/val/offsite)
# CSV format: id,D,G,A
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
        labels = torch.tensor(row.iloc[1:].values.astype("float32"))  # (3,)

        if self.transform:
            img = self.transform(img)
        return img, labels


# ========================
# Dataset: onsite (no labels)
# ========================
class RetinaOnsiteDataset(Dataset):
    def __init__(self, template_csv, image_dir, transform=None):
        self.df = pd.read_csv(template_csv)
        if "id" not in self.df.columns:
            raise ValueError(f'Template CSV must contain "id". Found: {list(self.df.columns)}')
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
# Metrics helper (per disease)
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


# ============================================================
# TASK 4 OPTION 4: Conditional VAE (CVAE) for augmentation
# ============================================================

class CVAE(nn.Module):
    """
    Simple Conditional VAE for RGB images.
    - Condition is 3-dim multi-label vector (D,G,A).
    - We embed condition -> feature map and concatenate to encoder input.
    - Decoder also gets condition embedded and concatenated.
    """

    def __init__(self, img_size=128, z_dim=128, cond_dim=3):
        super().__init__()
        self.img_size = img_size
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        # Condition embedding -> (C_cond, H, W)
        self.cond_channels = 8
        self.cond_to_map = nn.Sequential(
            nn.Linear(cond_dim, self.cond_channels * img_size * img_size),
            nn.ReLU(inplace=True),
        )

        # Encoder: (3 + cond_channels) x H x W -> latent
        in_ch = 3 + self.cond_channels
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1),  # 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),     # 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),    # 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),   # 8
            nn.ReLU(inplace=True),
        )
        self.enc_out_hw = img_size // 16  # for 128 => 8
        enc_flat = 256 * self.enc_out_hw * self.enc_out_hw

        self.fc_mu = nn.Linear(enc_flat, z_dim)
        self.fc_logvar = nn.Linear(enc_flat, z_dim)

        # Decoder: z + cond -> image
        self.fc_dec = nn.Sequential(
            nn.Linear(z_dim + cond_dim, enc_flat),
            nn.ReLU(inplace=True),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 128
            nn.Sigmoid(),  # output in [0,1]
        )

    def encode(self, x, y):
        b = x.size(0)
        cond_map = self.cond_to_map(y).view(b, self.cond_channels, self.img_size, self.img_size)
        x_in = torch.cat([x, cond_map], dim=1)
        h = self.enc(x_in).view(b, -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        zy = torch.cat([z, y], dim=1)
        h = self.fc_dec(zy)
        h = h.view(-1, 256, self.enc_out_hw, self.enc_out_hw)
        out = self.dec(h)
        return out

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    # recon loss (pixel BCE) + KL
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def train_cvae(
    train_csv="train.csv",
    train_image_dir="./images/train",
    img_size=128,
    z_dim=128,
    epochs=8,
    batch_size=32,
    lr=1e-3,
    save_path="checkpoints/task4_cvae.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # VAE train transform: resize + ToTensor in [0,1]
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = CVAE(img_size=img_size, z_dim=z_dim, cond_dim=3).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            recon, mu, logvar = model(x, y)
            loss = vae_loss(recon, x, mu, logvar)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg = total / len(ds)
        print(f"[CVAE] Epoch {ep}/{epochs} | Loss per sample: {avg:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved CVAE to: {save_path}")
    return save_path


def generate_augmented_images(
    cvae_ckpt,
    train_csv="train.csv",
    train_image_dir="./images/train",
    out_image_dir="./images/train_aug",
    out_csv="train_aug.csv",
    img_size=128,
    z_dim=128,
    # how many synthetic images to add for each minority condition
    n_glaucoma=200,
    n_amd=200,
):
    """
    Strategy:
    - Train set is imbalanced (DR >> G, AMD).
    - We generate extra samples conditioned on y=[*,1,*] (Glaucoma-positive)
      and y=[*,*,1] (AMD-positive), by sampling label vectors from real
      examples that contain those positives (keeps label combinations realistic).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(train_csv)

    # pick real label rows where G=1 or A=1
    df_g = df[df["G"] == 1].copy()
    df_a = df[df["A"] == 1].copy()

    if len(df_g) == 0 or len(df_a) == 0:
        raise ValueError("No samples found for Glaucoma or AMD in train.csv; cannot condition augmentation.")

    os.makedirs(out_image_dir, exist_ok=True)

    model = CVAE(img_size=img_size, z_dim=z_dim, cond_dim=3).to(device)
    model.load_state_dict(torch.load(cvae_ckpt, map_location=device))
    model.eval()

    def sample_and_generate(rows_df, n_samples, prefix):
        new_rows = []
        for i in range(n_samples):
            # sample an existing label pattern from that subset
            row = rows_df.sample(1).iloc[0]
            y = torch.tensor([[row["D"], row["G"], row["A"]]], dtype=torch.float32, device=device)

            z = torch.randn(1, z_dim, device=device)
            with torch.no_grad():
                x_gen = model.decode(z, y).cpu().squeeze(0)  # (3,H,W) in [0,1]

            # save as jpg
            img = transforms.ToPILImage()(x_gen)
            fname = f"{prefix}_syn_{i:05d}.jpg"
            fpath = os.path.join(out_image_dir, fname)
            img.save(fpath, quality=95)

            new_rows.append([fname, int(row["D"]), int(row["G"]), int(row["A"])])
        return new_rows

    print("Generating synthetic images for Glaucoma-positive patterns...")
    new_g = sample_and_generate(df_g, n_glaucoma, "G")

    print("Generating synthetic images for AMD-positive patterns...")
    new_a = sample_and_generate(df_a, n_amd, "A")

    # Build augmented CSV:
    # - original rows reference original train images
    # - synthetic rows reference train_aug directory
    # For training we will use a dataset that can read from TWO dirs.
    aug_df = df.copy()
    syn_df = pd.DataFrame(new_g + new_a, columns=["id", "D", "G", "A"])
    syn_df["is_synthetic"] = 1
    aug_df["is_synthetic"] = 0

    out_df = pd.concat([aug_df, syn_df], ignore_index=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved augmented CSV: {out_csv}")
    print(f"Augmented size: {len(out_df)} (original {len(df)} + synthetic {len(syn_df)})")

    return out_csv, out_image_dir


# ========================
# Augmented dataset loader (reads from train and train_aug)
# ========================
class RetinaAugmentedDataset(Dataset):
    """
    Reads a CSV that contains:
      id, D, G, A, is_synthetic
    If is_synthetic=0 -> image from train_dir
    If is_synthetic=1 -> image from syn_dir
    """
    def __init__(self, csv_file, train_dir, syn_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        if "is_synthetic" not in self.df.columns:
            raise ValueError("Augmented CSV must contain 'is_synthetic' column.")
        self.train_dir = train_dir
        self.syn_dir = syn_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["id"]
        label = torch.tensor([row["D"], row["G"], row["A"]], dtype=torch.float32)

        base_dir = self.syn_dir if int(row["is_synthetic"]) == 1 else self.train_dir
        img_path = os.path.join(base_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# Fine-tune ResNet18 on augmented data (Task1_3 style)
# ============================================================
def finetune_resnet18_on_augmented(
    train_aug_csv="train_aug.csv",
    train_dir="./images/train",
    syn_dir="./images/train_aug",
    val_csv="val.csv",
    offsite_csv="offsite_test.csv",
    val_dir="./images/val",
    offsite_dir="./images/offsite_test",
    onsite_template_csv="onsite_test_submission.csv",
    onsite_dir="./images/onsite_test",
    pretrained_ckpt="./pretrained_backbone/ckpt_resnet18_ep50.pt",
    save_dir="checkpoints",
    best_ckpt_name="task4_vae_aug_best_resnet18.pt",
    epochs=10,
    batch_size=16,
    lr=1e-5,
    img_size=256,
    threshold=0.5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(save_dir, exist_ok=True)

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

    # Datasets
    train_ds = RetinaAugmentedDataset(train_aug_csv, train_dir=train_dir, syn_dir=syn_dir, transform=train_tf)
    val_ds = RetinaMultiLabelDataset(val_csv, val_dir, transform=eval_tf)
    offsite_ds = RetinaMultiLabelDataset(offsite_csv, offsite_dir, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    offsite_loader = DataLoader(offsite_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    # Load pretrained weights
    state = torch.load(pretrained_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("\nLoaded pretrained ckpt with strict=False:")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    # Full fine-tuning
    for p in model.parameters():
        p.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_path = os.path.join(save_dir, best_ckpt_name)
    best_val_loss = float("inf")

    print("\nTask 4: Full fine-tuning on VAE-augmented dataset (train_aug.csv -> val.csv)")
    for ep in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_sum += loss.item() * x.size(0)

        train_loss = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_sum += loss.item() * x.size(0)

        val_loss = val_sum / len(val_loader.dataset)
        print(f"[resnet18_task4] Epoch {ep}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to: {best_path}")

    # OFFSITE evaluation
    print("\nEvaluating best Task 4 model on OFFSITE test set (labels available)")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for x, y in offsite_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            y_true_list.append(y.numpy())
            y_pred_list.append(preds)

    y_true_np = np.concatenate(y_true_list, axis=0)
    y_pred_np = np.concatenate(y_pred_list, axis=0)
    compute_and_print_metrics(y_true_np, y_pred_np, tag="resnet18_task4_vae_aug")

    # ONSITE submission
    print("\nGenerating Kaggle ONSITE submission CSV for Task 4...")
    onsite_ds = RetinaOnsiteDataset(onsite_template_csv, onsite_dir, transform=eval_tf)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_ids, all_preds = [], []
    with torch.no_grad():
        for x, img_names in onsite_loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_ids.extend(list(img_names))
            all_preds.extend(list(preds))

    template_df = pd.read_csv(onsite_template_csv)
    col_names = template_df.columns.tolist()
    disease_cols = col_names[1:]

    out_df = pd.DataFrame({"id": all_ids})
    out_df = pd.concat([out_df, pd.DataFrame(all_preds, columns=disease_cols)], axis=1)

    out_csv = "task4_vae_aug_onsite_submission.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(out_df.head())


# ============================================================
# Main runner
# ============================================================
if __name__ == "__main__":
    set_seed(42)

    # 1) Train CVAE (small size for feasibility)
    cvae_ckpt = train_cvae(
        train_csv="train.csv",
        train_image_dir="./images/train",
        img_size=128,
        z_dim=128,
        epochs=8,
        batch_size=32,
        lr=1e-3,
        save_path="checkpoints/task4_cvae.pt",
    )

    # 2) Generate synthetic images + augmented CSV
    train_aug_csv, syn_dir = generate_augmented_images(
        cvae_ckpt=cvae_ckpt,
        train_csv="train.csv",
        train_image_dir="./images/train",
        out_image_dir="./images/train_aug",
        out_csv="train_aug.csv",
        img_size=128,
        z_dim=128,
        n_glaucoma=200,
        n_amd=200,
    )

    # 3) Fine-tune ResNet18 on augmented dataset and create submission
    finetune_resnet18_on_augmented(
        train_aug_csv=train_aug_csv,
        train_dir="./images/train",
        syn_dir=syn_dir,
        val_csv="val.csv",
        offsite_csv="offsite_test.csv",
        val_dir="./images/val",
        offsite_dir="./images/offsite_test",
        onsite_template_csv="onsite_test_submission.csv",
        onsite_dir="./images/onsite_test",
        pretrained_ckpt="./pretrained_backbone/ckpt_resnet18_ep50.pt",
        save_dir="checkpoints",
        best_ckpt_name="task4_vae_aug_best_resnet18.pt",
        epochs=10,
        batch_size=16,
        lr=1e-5,
        img_size=256,
        threshold=0.5,
    )
