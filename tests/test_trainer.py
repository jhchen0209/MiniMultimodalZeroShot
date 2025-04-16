import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from utils.trainer import Trainer
from torch.utils.data import DataLoader

class MockVisionModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.dummy_layer = nn.Linear(768, 768).to(self.device)

    def forward(self, pixel_values):
        batch_size = pixel_values.size(0)
        dummy_input = torch.randn(batch_size, 768).to(self.device)
        return self.dummy_layer(dummy_input)  # [batch_size, 768]

class MockLLMModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.dummy_layer = nn.Linear(896, 32000).to(self.device)
        self.feature_projection = nn.Linear(32000, 896).to(self.device)  # 新增投影層
        self.tokenizer = MagicMock()
        self.tokenizer.pad_token_id = 0

    def forward(self, input_ids, attention_mask=None):
        # 處理3D輸入（來自forward的seq_output）
        if input_ids.dim() == 3:
            batch_size, seq_len, hidden_dim = input_ids.size()
            return self.dummy_layer(input_ids)  # [batch_size, seq_len, 32000]
        # 處理2D輸入（來自get_text_features的input_ids）
        else:
            batch_size, seq_len = input_ids.size()
            dummy_input = torch.randn(batch_size, seq_len, 896).half().to(self.device)
            return self.dummy_layer(dummy_input)  # [batch_size, seq_len, 32000]

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        self.dummy_param = nn.Parameter(torch.randn(1, 896))
        self.vision = MockVisionModule(self.device)
        self.llm = MockLLMModule(self.device)
        self.vision_to_llm = nn.Linear(768, 896).to(self.device)
        self.text_to_vision = nn.Linear(896, 768).to(self.device)

    def forward(self, pixel_values, input_ids, attention_mask):
        vision_features = self.vision(pixel_values)
        combined_features = self.vision_to_llm(vision_features)  # [batch_size, 896]
        batch_size, seq_len = input_ids.size()
        seq_output = combined_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, 896]
        return self.llm(seq_output)  # [batch_size, seq_len, 32000]

    def parameters(self):
        return [self.dummy_param, *self.vision.parameters(), *self.llm.parameters(),
                *self.vision_to_llm.parameters(), *self.text_to_vision.parameters()]

    def get_text_features(self, input_ids, attention_mask):
        llm_output = self.llm(input_ids, attention_mask)  # [batch_size, seq_len, 32000]
        text_features = llm_output.mean(dim=1)  # [batch_size, 32000]
        projected_features = self.llm.feature_projection(text_features)  # [batch_size, 896]
        return self.text_to_vision(projected_features)  # [batch_size, 768]

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.batch_size = 2
        self.seq_len = 32  # 新增序列長度
        
        # 模擬模型
        self.model = MockModel()
        
        # 模擬數據加載器
        self.train_dataloader = MagicMock(spec=DataLoader)
        self.val_dataloader = MagicMock(spec=DataLoader)
        
        # 模擬批次數據
        mock_batch = {
            "image": torch.randn(self.batch_size, 3, 224, 224).half().to(self.device),
            "text": torch.randint(0, 1000, (self.batch_size, self.seq_len)).to(self.device),
            "attention_mask": torch.ones(self.batch_size, self.seq_len).to(self.device)
        }
        self.train_dataloader.__iter__.return_value = iter([mock_batch])
        self.val_dataloader.__iter__.return_value = iter([mock_batch])
        
        # 模擬 len() 方法
        self.train_dataloader.__len__.return_value = 1
        self.val_dataloader.__len__.return_value = 1
        
        # 配置
        self.config = {
            "training": {
                "learning_rate": 1e-4,
                "epochs": 1,
                "batch_size": self.batch_size,
                "contrastive_weight": 0.5,
                "generation_weight": 0.5,
                "output_dir": "checkpoints"
            }
        }
        
        # 初始化 Trainer
        self.trainer = Trainer(self.model, self.train_dataloader, self.val_dataloader, self.config)

    def test_initialization(self):
        self.assertIsInstance(self.trainer.optimizer, torch.optim.AdamW)
        self.assertEqual(self.trainer.device, self.device)
        self.assertEqual(self.trainer.config["training"]["learning_rate"], 1e-4)
        self.assertEqual(self.trainer.config["training"]["epochs"], 1)

    def test_train_step(self):
        batch = next(iter(self.train_dataloader))
        loss, contrastive_loss, generation_loss = self.trainer.train_step(batch)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(contrastive_loss, float)
        self.assertIsInstance(generation_loss, float)

    def test_contrastive_loss(self):
        image_features = torch.randn(self.batch_size, 768).half().to(self.device)
        text_features = torch.randn(self.batch_size, 768).half().to(self.device)
        loss = self.trainer.contrastive_loss(image_features, text_features)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.dtype == torch.float16)

    def test_validate(self):
        val_loss = self.trainer.validate()
        self.assertIsInstance(val_loss, float)

if __name__ == '__main__':
    unittest.main()