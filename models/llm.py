# modesl/llm.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class QwenModel(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2-0.5B", use_lora=True, lora_rank=16):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            attn_implementation="sdpa",     # 注意力層實作(穩定)，sdpa:加速版本
            sliding_window=None             # image caption沒有長上下文需求
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if use_lora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            
    def generate(self, input_ids, max_length=128, **kwargs):
        return self.model.generate(
            input_ids=input_ids,        # iput_ids = Tokenize(text)
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,             # 啟用隨機採樣（否則模型會只挑最大機率的 token）
            top_p=0.9,                  # nucleus sampling（只在總機率為前 90% 的 token 中取樣）
            temperature=0.7,            # 溫度參數
            **kwargs                    # 可在呼叫時額外傳入如 eos_token_id 等參數
        )
    
    # Text -> Token
    def encode_text(self, text):
        return self.tokenizer(text, 
                              return_tensors="pt",  # 轉為Pytorch tensor
                              padding=True, 
                              truncation=True,      # 超過MAX Length截斷
                              max_length=128)