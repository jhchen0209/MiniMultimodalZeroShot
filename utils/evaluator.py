# utils/evaluator.py
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from models.multimodal import MultimodalModel
from utils.data_loader import MultimodalDataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Evaluator:
    def __init__(self, model, config, processor):
        self.model = model
        self.config = config
        self.processor = processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # CIFAR-10 類別
        self.cifar10_classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def evaluate_classification(self, dataloader):
        correct = 0
        total = 0

        # 預先編碼所有類別文本
        class_texts = [f"a photo of a {cls}" for cls in self.cifar10_classes]
        text_features = []
        
        # 批次處理類別文本特徵
        with torch.no_grad():
            # 使用 tokenizer 處理文本
            text_inputs = self.model.llm.tokenizer(
                class_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 獲取文本特徵
            text_features = self.model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"]
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 批次處理圖像
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Classification"):
                images = batch["image"].to(self.device)
                labels = batch["text"]
                
                # 獲取圖像特徵
                image_features = self.model.vision(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 計算相似度
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                predicted_labels = similarity.argmax(dim=-1)
                
                # 計算正確預測數
                for i, pred in enumerate(predicted_labels):
                    if self.cifar10_classes[pred] == labels[i]:
                        correct += 1
                total += len(labels)

        return correct / total if total > 0 else 0.0

    def evaluate_description(self, dataloader):
        bleu_scores = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating Description"):
                images = batch["image"].to(self.device)
                reference_captions = batch["text"]  # 這應該是一個字符串列表

                # 生成描述
                outputs = self.model.generate(
                    pixel_values=images,
                    prompt="Describe this image.",
                    max_length=128
                )

                # 計算 BLEU 分數
                for gen_text, ref_text in zip(outputs, reference_captions):
                    # ref_text 應該已經是字符串，如果不是則轉換
                    if not isinstance(ref_text, str):
                        # 如果是列表或其他序列類型，取第一個元素
                        if hasattr(ref_text, '__len__'):
                            ref_text = ref_text[0]
                        ref_text = str(ref_text)
                    
                    # 分割文本為單詞列表
                    reference = ref_text.split()
                    candidate = gen_text.split()
                    
                    # 計算 BLEU 分數
                    try:
                        bleu = sentence_bleu(
                            [reference],
                            candidate,
                            weights=(0.25, 0.25, 0.25, 0.25),
                            smoothing_function=SmoothingFunction().method4
                        )
                        bleu_scores.append(bleu)
                    except Exception as e:
                        print(f"Warning: BLEU score calculation failed for a sample. Error: {e}")
                        print(f"Reference: {reference}")
                        print(f"Candidate: {candidate}")
                        continue

        # 返回平均 BLEU 分數
        return np.mean(bleu_scores) if bleu_scores else 0.0

    def run_evaluation(self):
        test_dataset = MultimodalDataset(
            flickr8k_path=self.config["data"]["flickr8k_path"],
            cifar10_path=self.config["data"]["cifar10_path"],
            processor=self.processor,
            split="val",
            max_samples=self.config.get("data", {}).get("max_samples", 5000)
        )

        # 分離 CIFAR-10 和 Flickr8k 數據
        cifar10_indices = [i for i in range(len(test_dataset)) if i >= test_dataset.flickr8k_size]
        flickr8k_indices = [i for i in range(len(test_dataset)) if i < test_dataset.flickr8k_size]

        # 創建數據加載器
        cifar10_dataset = torch.utils.data.Subset(test_dataset, cifar10_indices)
        flickr8k_dataset = torch.utils.data.Subset(test_dataset, flickr8k_indices)

        dataloaders = {
            'cifar10': DataLoader(
                cifar10_dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ),
            'flickr8k': DataLoader(
                flickr8k_dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        }

        # 評估
        classification_accuracy = self.evaluate_classification(dataloaders['cifar10'])
        description_bleu = self.evaluate_description(dataloaders['flickr8k'])

        return {
            "classification_accuracy": classification_accuracy,
            "description_bleu": description_bleu
        }