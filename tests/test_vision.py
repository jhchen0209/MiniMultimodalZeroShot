import unittest
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from models.vision import CLIPVisionModel

class MockModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        # 模擬 forward 方法的行為
        output = MagicMock()
        output.pooler_output = torch.randn(2, 768)
        return output

class TestCLIPVisionModel(unittest.TestCase):
    def setUp(self):
        self.model = CLIPVisionModel(model_name="openai/clip-vit-base-patch32")

    @patch('models.vision.AutoModel')
    @patch('models.vision.AutoProcessor')
    def test_initialization(self, mock_processor, mock_model):
        # 模擬模型和處理器
        mock_model_instance = MockModule()
        mock_processor_instance = MagicMock()
        mock_model.from_pretrained.return_value.vision_model = mock_model_instance
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        # 初始化模型
        model = CLIPVisionModel()
        
        # 驗證模型和處理器
        self.assertEqual(model.model, mock_model_instance)
        self.assertEqual(model.processor, mock_processor_instance)
        
        # 驗證參數是否被凍結
        for param in mock_model_instance.parameters():
            self.assertFalse(param.requires_grad)

    def test_forward(self):
        pixel_values = torch.randn(2, 3, 224, 224).half()
        
        # 使用 MockModule 模擬模型
        self.model.model = MockModule()
        
        # 執行 forward
        output = self.model(pixel_values)
        
        # 驗證輸出
        self.assertEqual(output.shape, (2, 768))

    def test_preprocess_image(self):
        images = [torch.randn(3, 224, 224)] * 2
        self.model.processor = MagicMock(return_value={"pixel_values": torch.randn(2, 3, 224, 224)})
        
        # 執行預處理
        output = self.model.preprocess_image(images)
        
        # 驗證調用和輸出
        self.model.processor.assert_called_once_with(images=images, return_tensors="pt", padding=True)
        self.assertEqual(output.shape, (2, 3, 224, 224))

if __name__ == '__main__':
    unittest.main()