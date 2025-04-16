import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from models.multimodal import MultimodalModel

class MockVisionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")

    def forward(self, *args, **kwargs):
        return torch.randn(2, 768).half().to(self.device)

class MockLLMModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")
        self.config = MagicMock(hidden_size=896)
        self.embedding = nn.Embedding(32000, 896).half().to(self.device)

    def forward(self, *args, **kwargs):
        output = MagicMock()
        output.logits = torch.randn(2, 33, 32000).half().to(self.device)
        output.hidden_states = [None, torch.randn(2, 32, 896).half().to(self.device)]
        return output

    def get_input_embeddings(self):
        return self.embedding

class TestMultimodalModel(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.model = MultimodalModel(
            llm_model_name="Qwen/Qwen2-0.5B",
            vision_model_name="openai/clip-vit-base-patch32",
            lora_rank=16
        ).to(self.device)  # 確保模型在 CUDA 上
        self.batch_size = 2
        self.pixel_values = torch.randn(self.batch_size, 3, 224, 224).half().to(self.device)
        self.input_ids = torch.randint(0, 1000, (self.batch_size, 32)).to(self.device)
        self.attention_mask = torch.ones(self.batch_size, 32).to(self.device)

    @patch('models.multimodal.CLIPVisionModel')
    @patch('models.multimodal.QwenModel')
    def test_initialization(self, mock_qwen, mock_clip):
        mock_qwen_instance = MagicMock()
        mock_clip_instance = MagicMock()
        mock_qwen.return_value = mock_qwen_instance
        mock_clip.return_value = mock_clip_instance
        model = MultimodalModel().to(self.device)
        self.assertIsInstance(model.vision_to_llm, nn.Linear)
        self.assertIsInstance(model.text_to_vision, nn.Linear)
        self.assertEqual(model.vision_to_llm.out_features, mock_qwen_instance.model.config.hidden_size)

    def test_forward(self):
        # 模擬 vision 和 llm 的行為
        self.model.vision = MockVisionModule()
        self.model.llm.model = MockLLMModule()
        
        # 執行 forward
        logits = self.model(self.pixel_values, self.input_ids, self.attention_mask)
        
        # 驗證輸出
        self.assertEqual(logits.shape, (self.batch_size, 33, 32000))

    def test_generate(self):
        # 模擬 vision 和 llm 的行為
        self.model.vision = MockVisionModule()
        self.model.llm.model = MockLLMModule()
        self.model.llm.model.generate = MagicMock(return_value=torch.randint(0, 1000, (self.batch_size, 128)).to(self.device))
        self.model.llm.tokenizer.batch_decode = MagicMock(return_value=["Generated text"] * self.batch_size)
        self.model.llm.encode_text = MagicMock(return_value={
            "input_ids": torch.randint(0, 1000, (self.batch_size, 32)).to(self.device),
            "attention_mask": torch.ones(self.batch_size, 32).to(self.device)
        })
        
        # 執行 generate
        outputs = self.model.generate(self.pixel_values, prompt="Describe this image.")
        
        # 驗證輸出
        self.assertEqual(len(outputs), self.batch_size)
        self.assertTrue(all(isinstance(out, str) for out in outputs))

    def test_preprocess(self):
        images = [torch.randn(3, 224, 224)] * self.batch_size
        texts = ["Test caption"] * self.batch_size
        self.model.vision.preprocess_image = MagicMock(return_value=torch.randn(self.batch_size, 3, 224, 224).to(self.device))
        self.model.llm.encode_text = MagicMock(return_value={
            "input_ids": torch.randint(0, 1000, (self.batch_size, 128)).to(self.device),
            "attention_mask": torch.ones(self.batch_size, 128).to(self.device)
        })
        result = self.model.preprocess(images, texts)
        self.assertIn("pixel_values", result)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

    def test_get_text_features(self):
        # 模擬 llm.model 的行為
        self.model.llm.model = MockLLMModule()
        
        # 執行 get_text_features
        text_features = self.model.get_text_features(self.input_ids, self.attention_mask)
        
        # 驗證輸出
        self.assertEqual(text_features.shape, (self.batch_size, 768))

if __name__ == '__main__':
    unittest.main()