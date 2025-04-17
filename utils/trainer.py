import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import os
from models.multimodal import MultimodalModel
from torch.optim.lr_scheduler import LambdaLR

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
        self.model.to(self.device)

        learning_rate = float(config["training"]["learning_rate"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.total_epochs = config["training"]["epochs"]
        self.total_steps = len(train_dataloader) * self.total_epochs
        self.warmup_steps = int(self.total_steps * config["training"].get("warmup_ratio", 0.1))

        # linear decay schedule
        self.scheduler = self.get_scheduler()

        self.scaler = GradScaler()
        self.contrastive_weight = config.get("training", {}).get("contrastive_weight", 0.5)
        self.generation_weight = config.get("training", {}).get("generation_weight", 0.5)
        self.output_dir = config.get("training", {}).get("output_dir", "checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_scheduler(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0, float(self.total_steps - current_step) / float(max(1, self.total_steps - self.warmup_steps))
            )
        return LambdaLR(self.optimizer, lr_lambda)
    
    def contrastive_loss(self, vision_features, text_features, temperature=0.07):
        vision_features = F.normalize(vision_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = torch.matmul(vision_features, text_features.t()) / temperature
        labels = torch.arange(logits.size(0)).to(self.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

    def train_step(self, batch):
        pixel_values = batch["image"].to(self.device)
        input_ids = batch["text"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        self.optimizer.zero_grad()

        with autocast(device_type=self.device.type, enabled=self.use_amp):
            # 提取視覺特徵
            vision_features = self.model.vision(pixel_values)
            # 提取文字特徵
            text_features = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 對比損失
            contrastive_loss = self.contrastive_loss(vision_features, text_features)

            # 生成損失
            logits = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            shift_logits = logits[:, 1:-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            #print("logits:", shift_logits.shape)   # 預期 (B, T-1, V)
            #print("labels:", shift_labels.shape)   # 預期 (B, T-1)

            generation_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=self.model.llm.tokenizer.pad_token_id
            )


            total_loss = (
                self.contrastive_weight * contrastive_loss +
                self.generation_weight * generation_loss
            )
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return total_loss.item(), contrastive_loss.item(), generation_loss.item()

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                pixel_values = batch["image"].to(self.device)
                input_ids = batch["text"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    vision_features = self.model.vision(pixel_values)
                    text_features = self.model.get_text_features(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    contrastive_loss = self.contrastive_loss(vision_features, text_features)

                    logits = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    shift_logits = logits[:, 1:-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()
                    generation_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=self.model.llm.tokenizer.pad_token_id
                    )

                    total_loss += (
                        self.contrastive_weight * contrastive_loss +
                        self.generation_weight * generation_loss
                    ).item()

        avg_loss = total_loss / len(self.val_dataloader)
        self.model.train()
        return avg_loss

    def train(self):
        epochs = self.config["training"]["epochs"]
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            total_loss, total_contrastive, total_generation = 0.0, 0.0, 0.0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                loss, contrastive_loss, generation_loss = self.train_step(batch)
                total_loss += loss
                total_contrastive += contrastive_loss
                total_generation += generation_loss

            avg_loss = total_loss / len(self.train_dataloader)
            avg_contrastive = total_contrastive / len(self.train_dataloader)
            avg_generation = total_generation / len(self.train_dataloader)

            val_loss = self.validate()

            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {avg_loss:.4f} "
                  f"(Contrastive: {avg_contrastive:.4f}, Generation: {avg_generation:.4f}), "
                  f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss
                }, os.path.join(self.output_dir, "best_model.pth"))
                print(f"Saved best model at epoch {epoch + 1}")

        print("Training completed!")