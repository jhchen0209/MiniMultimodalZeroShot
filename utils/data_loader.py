# utils/data_loader.py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

class Flickr30kDataset(Dataset):
    def __init__(self, flickr30k_path, processor, split="train", max_samples=30000):
        self.processor = processor
        self.split = split
        self.max_samples = max_samples

        # 載入 Flickr30k
        self.data = self.load_flickr30k(flickr30k_path)
        self.total_size = min(len(self.data))

    def load_flickr30k(self, flickr30k_path):
        captions_file = os.path.join(flickr30k_path, "results.csv")
        df = pd.read_csv(captions_file)

        if self.split == "train":
            df = df.sample(frac=0.8, random_state=42)
        else:
            df = df.drop(df.sample(frac=0.8, random_state=42).index)

        df["image_path"] = df["image"].apply(lambda x: os.path.join(flickr30k_path, "flickr30k_images", x))
        return df.reset_index(drop=True)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        image_path = item["image_path"]
        caption = item["caption"]

        # 載入圖像
        image = Image.open(image_path).convert("RGB")

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

    train_dataset = Flickr30kDataset(
        flickr30k_path=config["data"]["flickr30k_path"],
        processor=processor,
        split="train",
        max_samples=config.get("data", {}).get("max_samples", 30000)
    )

    val_dataset = Flickr30kDataset(
        flickr30k_path=config["data"]["flickr30k_path"],
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
