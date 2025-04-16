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

    # 訓練後評估
    print("Evaluating model...")
    evaluator = Evaluator(
        model=model,
        config=config,
        processor=model.vision.processor
    )
    results = evaluator.run_evaluation()
    print(f"Classification Accuracy: {results['classification_accuracy']:.4f}")
    print(f"Description BLEU Score: {results['description_bleu']:.4f}")

if __name__ == "__main__":
    main()