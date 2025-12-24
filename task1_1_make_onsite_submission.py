import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ------------------------
# Dataset for ON-SITE test (no labels)
# Reads the official template CSV to get image ids and required columns.
# ------------------------
class RetinaOnsiteDataset(Dataset):
    def __init__(self, template_df, image_dir, transform=None):
        self.df = template_df.reset_index(drop=True)

        if "id" not in self.df.columns:
            raise ValueError(
                f'Column "id" not found. Found columns: {list(self.df.columns)}.\n'
                "Use the original onsite_test_submission.csv and do NOT edit it in Excel."
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


def build_resnet18(num_classes=3):
    model = models.resnet18(weights=None)  # avoids deprecated pretrained arg
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_onsite_submission(
    template_csv="onsite_test_submission.csv",
    image_dir="./images/onsite_test",
    ckpt_path="./pretrained_backbone/ckpt_resnet18_ep50.pt",
    output_csv="task1_1_onsite_submission.csv",
    batch_size=32,
    img_size=256,
    threshold=0.5,
):
    # 1) Read template and detect required label columns (whatever Kaggle expects)
    template_df = pd.read_csv(template_csv)

    if "id" not in template_df.columns:
        raise ValueError(f'Template file must contain "id". Found: {list(template_df.columns)}')

    label_cols = [c for c in template_df.columns if c != "id"]
    if len(label_cols) != 3:
        raise ValueError(
            f"Expected exactly 3 label columns besides 'id', but found {len(label_cols)}: {label_cols}\n"
            "This must match the provided competition template."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = RetinaOnsiteDataset(template_df, image_dir, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)  # Windows: num_workers=0

    # 2) Build model
    model = build_resnet18(num_classes=3).to(device)

    # 3) Load checkpoint
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 4) Predict
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for imgs, img_names in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)  # 0/1 predictions

            all_ids.extend(list(img_names))
            all_preds.extend(list(preds))

    # 5) Write output using EXACT template columns
    pred_df = pd.DataFrame(all_preds, columns=label_cols)
    out_df = pd.DataFrame({"id": all_ids})
    out_df = pd.concat([out_df, pred_df], axis=1)

    # Final hard checks
    if out_df.columns.tolist() != ["id"] + label_cols:
        raise ValueError(f"Wrong columns. Got {out_df.columns.tolist()}, expected {['id'] + label_cols}")

    if len(out_df) != len(template_df):
        raise ValueError(f"Row count mismatch. Got {len(out_df)}, expected {len(template_df)}")

    # Make sure values are integers
    for c in label_cols:
        out_df[c] = out_df[c].astype(int)

    out_df.to_csv(output_csv, index=False)
    print(f"Saved Kaggle submission file: {output_csv}")
    print("Columns:", out_df.columns.tolist())
    print(out_df.head())


if __name__ == "__main__":
    make_onsite_submission()
