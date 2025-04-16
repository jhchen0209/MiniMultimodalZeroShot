# scripts/test.py
import yaml
import torch
import argparse
from models.multimodal import MultimodalModel
from utils.evaluator import Evaluator

def main(args):
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

    # 載入訓練好的權重
    checkpoint_path = args.checkpoint or "checkpoints/best_model.pth"
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device("cuda"))

    # 初始化評估器
    print("Setting up evaluator...")
    evaluator = Evaluator(
        model=model,
        config=config,
        processor=model.vision.processor
    )

    # 運行評估
    print("Running evaluation...")
    results = evaluator.run_evaluation()
    print(f"Classification Accuracy: {results['classification_accuracy']:.4f}")
    print(f"Description BLEU Score: {results['description_bleu']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained multimodal model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to the trained model checkpoint"
    )
    args = parser.parse_args()
    main(args)