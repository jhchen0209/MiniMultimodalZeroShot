import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class QwenModel(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2-0.5B", use_lora=True, lora_rank=16):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
            sliding_window=None
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
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            **kwargs
        )
    
    def encode_text(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)