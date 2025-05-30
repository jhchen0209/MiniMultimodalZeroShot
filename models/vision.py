# models/vision.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model

class CLIPVisionModel(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).vision_model
        
        # 多模態前處理
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=True
        )
            
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.pooler_output    # 經過池化操作後的視覺特徵向量

    def preprocess_image(self, images):
        return self.processor(images=images, return_tensors="pt", padding=True)["pixel_values"]