# scripts/train.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from models.multimodal import MultimodalModel
from utils.data_loader import get_dataloaders
from utils.trainer import Trainer
from utils.evaluator import Evaluator

def main():
    # 載入配置文件
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 初始化模型
    print("Initializing model...")
    model = MultimodalModel(
        llm_model_name=config["model"]["llm"],
        vision_model_name=config["model"]["vision"],
        lora_rank=config["model"]["lora_rank"]
    )

    # 獲取數據載入器
    print("Loading data...")
    train_dataloader, val_dataloader = get_dataloaders(
        config=config,
        processor=model.vision.processor
    )

    # 初始化訓練器
    print("Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )

    # 開始訓練
    print("Starting training...")
    trainer.train()

    print("Training done")

if __name__ == "__main__":
    main()
