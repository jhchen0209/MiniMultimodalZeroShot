# models/multimodal.py
import torch
import torch.nn as nn
from .llm import QwenModel
from .vision import CLIPVisionModel

class MultimodalModel(nn.Module):
    def __init__(self, llm_model_name="Qwen/Qwen2-0.5B", vision_model_name="openai/clip-vit-base-patch32", lora_rank=16):
        super(MultimodalModel, self).__init__()
        self.llm = QwenModel(model_name=llm_model_name, use_lora=True, lora_rank=lora_rank)
        self.vision = CLIPVisionModel(model_name=vision_model_name)

        # 將 CLIP 輸出的 768 維向量轉換為 Qwen LLM 的隱藏維度（896）。
        self.vision_to_llm = nn.Linear(768, self.llm.model.config.hidden_size)

        # 將 Qwen 的輸出轉成 768 維，對齊視覺空間，用在圖文對比任務。
        self.text_to_vision = nn.Linear(self.llm.model.config.hidden_size, 768)
        self.dropout = nn.Dropout(0.1)
    
    # 圖文對齊
    def forward(self, pixel_values, input_ids, attention_mask=None):
        vision_features = self.vision(pixel_values)
        vision_features = self.vision_to_llm(vision_features)
        vision_features = self.dropout(vision_features)

        batch_size = input_ids.size(0)
        vision_token = vision_features.unsqueeze(1)
        embeddings = self.llm.model.get_input_embeddings()(input_ids) # 文本轉 embedding
        combined_embeddings = torch.cat([vision_token, embeddings], dim=1)

        if attention_mask is not None:
            vision_mask = torch.ones(batch_size, 1, device=input_ids.device)
            combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        # 傳入合併後的 embedding
        outputs = self.llm.model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            return_dict=True
        )
        return outputs.logits

    def generate(self, pixel_values, prompt=None, max_length=128):
        device = self.llm.model.device
        batch_size = pixel_values.size(0)  # 獲取批次大小
        
        # 處理視覺特徵
        pixel_values = pixel_values.to(device=device, dtype=torch.float16)
        with torch.amp.autocast('cuda'):  
            vision_features = self.vision(pixel_values)
            vision_features = vision_features.to(dtype=torch.float16)
            vision_features = self.vision_to_llm(vision_features)
            vision_features = self.dropout(vision_features).unsqueeze(1)

            if prompt:
                # 處理文本輸入，確保批次大小匹配
                inputs = self.llm.encode_text([prompt] * batch_size)  # 為每個圖像複製相同的提示
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                with torch.amp.autocast('cuda'):
                    embeddings = self.llm.model.get_input_embeddings()(input_ids)
                    combined_embeddings = torch.cat([vision_features, embeddings], dim=1)
                    
                    # 為視覺特徵添加 attention mask
                    vision_mask = torch.ones((batch_size, 1), device=device)
                    combined_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
            else:
                combined_embeddings = vision_features
                combined_attention_mask = torch.ones((batch_size, 1), device=device)

            # 生成文本
            outputs = self.llm.model.generate(
                inputs_embeds=combined_embeddings,
                attention_mask=combined_attention_mask,
                pad_token_id=self.llm.tokenizer.pad_token_id or self.llm.tokenizer.eos_token_id,
                max_new_tokens=max_length,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        
        return self.llm.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def preprocess(self, images, texts=None):
        pixel_values = self.vision.preprocess_image(images).to(self.llm.model.device)
        if texts:
            text_inputs = self.llm.encode_text(texts)
            return {
                "pixel_values": pixel_values,
                "input_ids": text_inputs["input_ids"].to(self.llm.model.device),
                "attention_mask": text_inputs["attention_mask"].to(self.llm.model.device)
            }
        return {"pixel_values": pixel_values}

    def get_text_features(self, input_ids, attention_mask=None):
        # 提取文字特徵並映射到視覺特徵空間
        text_outputs = self.llm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        text_features = text_outputs.hidden_states[-1][:, -1, :]  # [batch, 896]
        # 映射到視覺特徵空間
        text_features = self.text_to_vision(text_features)  # [batch, 768]
        
        return text_features