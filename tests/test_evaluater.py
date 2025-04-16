import unittest
from unittest.mock import patch, MagicMock
import torch
from utils.evaluator import Evaluator
from torch.utils.data import DataLoader

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.config = {
            "data": {
                "flickr8k_path": "data/flickr8k",
                "cifar10_path": "data/cifar10",
                "max_samples": 5000
            },
            "training": {
                "batch_size": 16
            }
        }
        self.model = MagicMock()
        self.processor = MagicMock()
        self.evaluator = Evaluator(self.model, self.config, self.processor)

    def test_evaluate_classification(self):
        dataloader = MagicMock()
        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "text": ["cat", "dog"]
        }
        dataloader.__iter__.return_value = [batch]
        self.model.vision = MagicMock(return_value=torch.randn(2, 768))
        self.model.get_text_features = MagicMock(return_value=torch.randn(2, 768))
        self.model.llm.tokenizer = MagicMock(return_value={
            "input_ids": torch.randint(0, 1000, (10, 32)),
            "attention_mask": torch.ones(10, 32)
        })
        accuracy = self.evaluator.evaluate_classification(dataloader)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0.0 <= accuracy <= 1.0)

    def test_evaluate_description(self):
        dataloader = MagicMock()
        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "text": ["A cat sitting", "A dog running"]
        }
        dataloader.__iter__.return_value = [batch]
        self.model.generate = MagicMock(return_value=["Generated cat", "Generated dog"])
        bleu_score = self.evaluator.evaluate_description(dataloader)
        self.assertIsInstance(bleu_score, float)
        self.assertTrue(0.0 <= bleu_score <= 1.0)

    @patch('utils.evaluator.MultimodalDataset')
    @patch('torch.utils.data.DataLoader')
    def test_run_evaluation(self, mock_dataloader, mock_dataset):
        mock_dataset.return_value = MagicMock()
        mock_dataloader.side_effect = [MagicMock(), MagicMock()]
        self.evaluator.evaluate_classification = MagicMock(return_value=0.85)
        self.evaluator.evaluate_description = MagicMock(return_value=0.65)
        results = self.evaluator.run_evaluation()
        self.assertIn("classification_accuracy", results)
        self.assertIn("description_bleu", results)
        self.assertEqual(results["classification_accuracy"], 0.85)
        self.assertEqual(results["description_bleu"], 0.65)

if __name__ == '__main__':
    unittest.main()