import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from utils.trainer import Trainer
from torch.utils.data import DataLoader

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        # 模擬可訓練參數
        self.dummy_param = nn.Parameter(torch.randn(1, 896))
        self.vision_to_llm = nn.Linear(768, 896).half().to(self.device)
        self.text_to_vision = nn.Linear(896, 768).half().to(self.device)

    def forward(self, pixel_values, input_ids, attention_mask):
        # 模擬 forward 輸出
        return torch.randn(2, 33, 32000).half().to(self.device)

    def parameters(self):
        # 返回可訓練參數
        return [self.dummy_param, *self.vision_to_llm.parameters(), *self.text_to_vision.parameters()]

    def get_text_features(self, input_ids, attention_mask):
        # 模擬 get_text_features 輸出
        return torch.randn(2, 768).half().to(self.device)

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.batch_size = 2
        
        # 模擬模型
        self.model = MockModel()
        
        # 模擬數據加載器
        self.train_dataloader = MagicMock(spec=DataLoader)
        self.val_dataloader = MagicMock(spec=DataLoader)
        
        # 模擬批次數據，使用 "image" 和 "text" 鍵以匹配 trainer.py
        mock_batch = {
            "image": torch.randn(self.batch_size, 3, 224, 224).half().to(self.device),
            "text": torch.randint(0, 1000, (self.batch_size, 32)).to(self.device),
            "attention_mask": torch.ones(self.batch_size, 32).to(self.device)
        }
        self.train_dataloader.__iter__.return_value = iter([mock_batch])
        self.val_dataloader.__iter__.return_value = iter([mock_batch])
        
        # 配置，匹配 trainer.py 期望的嵌套結構
        self.config = {
            "training": {
                "learning_rate": 1e-4,
                "num_epochs": 1,
                "batch_size": self.batch_size
            },
            "device": self.device
        }
        
        # 初始化 Trainer
        self.trainer = Trainer(self.model, self.train_dataloader, self.val_dataloader, self.config)

    def test_initialization(self):
        self.assertIsInstance(self.trainer.optimizer, torch.optim.AdamW)
        self.assertEqual(self.trainer.device, self.device)
        self.assertEqual(self.trainer.config["training"]["learning_rate"], 1e-4)
        self.assertEqual(self.trainer.config["training"]["num_epochs"], 1)

    def test_train_step(self):
        # 模擬模型輸出
        self.model.return_value = torch.randn(self.batch_size, 33, 32000).half().to(self.device)
        
        # 執行 train_step
        batch = next(iter(self.train_dataloader))
        loss = self.trainer.train_step(batch)
        
        # 驗證損失
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.dtype == torch.float16)

    def test_contrastive_loss(self):
        # 模擬特徵
        image_features = torch.randn(self.batch_size, 768).half().to(self.device)
        text_features = torch.randn(self.batch_size, 768).half().to(self.device)
        
        # 計算對比損失
        loss = self.trainer.contrastive_loss(image_features, text_features)
        
        # 驗證損失
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.dtype == torch.float16)

    def test_validate(self):
        # 模擬驗證過程
        val_loss = self.trainer.validate()
        
        # 驗證損失
        self.assertIsInstance(val_loss, float)

if __name__ == '__main__':
    unittest.main()