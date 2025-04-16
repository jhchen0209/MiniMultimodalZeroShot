# utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from PIL import Image
import pandas as pd
import os
from transformers import AutoProcessor
import numpy as np

class MultimodalDataset(Dataset):
    def __init__(self, flickr8k_path, cifar10_path, processor, split="train", max_samples=20000):
        self.processor = processor
        self.split = split
        self.max_samples = max_samples

        # 載入 Flickr8k
        self.flickr8k_data = self.load_flickr8k(flickr8k_path)
        # 載入 CIFAR-10
        self.cifar10_data = self.load_cifar10(cifar10_path)

        # 計算數據量
        self.flickr8k_size = len(self.flickr8k_data)
        self.cifar10_size = min(len(self.cifar10_data), 12000)
        self.total_size = min(self.flickr8k_size + self.cifar10_size, max_samples)

    def load_flickr8k(self, flickr8k_path):
        captions_file = os.path.join(flickr8k_path, "captions.txt")
        df = pd.read_csv(captions_file, names=["image", "caption"], skiprows=1)

        if self.split == "train":
            df = df.sample(frac=0.8, random_state=42)
        else:
            df = df.drop(df.sample(frac=0.8, random_state=42).index)

        df["image_path"] = df["image"].apply(lambda x: os.path.join(flickr8k_path, "Images", x))
        return df.reset_index(drop=True)

    def load_cifar10(self, cifar10_path):
        dataset = CIFAR10(
            root=cifar10_path,
            train=(self.split == "train"),
            download=False
        )
        return dataset

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if idx < self.flickr8k_size:
            # 從 Flickr8k 獲取樣本
            item = self.flickr8k_data.iloc[idx]
            image_path = item["image_path"]
            caption = item["caption"]
            # 載入 Flickr8k 圖像
            image = Image.open(image_path).convert("RGB")
        else:
            # 從 CIFAR-10 獲取樣本
            cifar_idx = idx - self.flickr8k_size
            image, label = self.cifar10_data[cifar_idx]
            caption = f"This is a picture of a {self.cifar10_data.classes[label]}"
            # 確保 image 是 NumPy 數組
            if isinstance(image, Image.Image):
                image = np.array(image)
            # 轉換為 PIL Image
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

        # 使用 CLIP 處理器預處理
        image_tensor = self.processor(
            images=image,
            return_tensors="pt",
            padding=True
        )["pixel_values"].squeeze(0)

        text_inputs = self.processor(
            text=caption,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True
        )
        input_ids = text_inputs["input_ids"].squeeze(0)
        attention_mask = text_inputs["attention_mask"].squeeze(0)

        return {
            "image": image_tensor,
            "text": input_ids,
            "attention_mask": attention_mask
        }

def get_dataloaders(config, processor):
    batch_size = config["training"]["batch_size"]
    num_workers = config.get("dataloader", {}).get("num_workers", 0) 
    
    train_dataset = MultimodalDataset(
        flickr8k_path=config["data"]["flickr8k_path"],
        cifar10_path=config["data"]["cifar10_path"],
        processor=processor,
        split="train",
        max_samples=config.get("data", {}).get("max_samples", 20000)
    )

    val_dataset = MultimodalDataset(
        flickr8k_path=config["data"]["flickr8k_path"],
        cifar10_path=config["data"]["cifar10_path"],
        processor=processor,
        split="val",
        max_samples=config.get("data", {}).get("max_samples", 5000)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader