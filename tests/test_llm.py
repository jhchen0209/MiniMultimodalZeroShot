import unittest
from unittest.mock import patch, MagicMock
import torch
from models.llm import QwenModel

class TestQwenModel(unittest.TestCase):
    def setUp(self):
        self.model = QwenModel(model_name="Qwen/Qwen2-0.5B", use_lora=True, lora_rank=16)

    @patch('models.llm.AutoModelForCausalLM')
    @patch('models.llm.AutoTokenizer')
    @patch('models.llm.get_peft_model')
    def test_initialization(self, mock_peft, mock_tokenizer, mock_model):
        mock_model_instance = MagicMock()
        mock_tokenizer_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        model = QwenModel(use_lora=True, lora_rank=16)
        mock_peft.assert_called_once()
        self.assertEqual(model.model, mock_peft.return_value)
        self.assertEqual(model.tokenizer, mock_tokenizer_instance)

    def test_generate(self):
        input_ids = torch.randint(0, 1000, (2, 32))
        self.model.model.generate = MagicMock(return_value=torch.randint(0, 1000, (2, 128)))
        outputs = self.model.generate(input_ids, max_length=128)
        self.model.model.generate.assert_called_once()
        self.assertEqual(outputs.shape, (2, 128))

    def test_encode_text(self):
        texts = ["Test text", "Another test"]
        self.model.tokenizer = MagicMock(return_value={
            "input_ids": torch.randint(0, 1000, (2, 128)),
            "attention_mask": torch.ones(2, 128)
        })
        result = self.model.encode_text(texts)
        self.model.tokenizer.assert_called_once_with(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        self.assertIn("input_ids", result)
        self.assertIn("attention_mask", result)

if __name__ == '__main__':
    unittest.main()