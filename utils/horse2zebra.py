import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


class HorseZebraDatasetFromCSV(Dataset):
    def __init__(self, root_dir, metadata_path, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Đọc metadata
        df = pd.read_csv(metadata_path)

        # Lọc theo split
        self.df = df[df["split"] == split].reset_index(drop=True)

        # Chuyển domain thành label (A = 0, B = 1)
        self.df["label"] = self.df["domain"].apply(lambda x: 0 if "A" in x else 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label
