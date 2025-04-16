import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from utils.data_loader import MultimodalDataset, get_dataloaders
from PIL import Image

class TestMultimodalDataset(unittest.TestCase):
    def setUp(self):
        self.config = {
            "data": {
                "flickr8k_path": "data/flickr8k",
                "cifar10_path": "data/cifar10",
                "max_samples": 20000
            },
            "training": {
                "batch_size": 16
            }
        }
        self.processor = MagicMock()

    @patch('utils.data_loader.pd.read_csv')
    @patch('utils.data_loader.CIFAR10')
    def test_dataset_initialization(self, mock_cifar, mock_csv):
        mock_csv.return_value = pd.DataFrame({
            "image": ["img1.jpg", "img2.jpg"],
            "caption": ["Caption 1", "Caption 2"]
        })
        mock_cifar.return_value = MagicMock(__len__=lambda x: 1000, classes=["cat", "dog"])
        dataset = MultimodalDataset(
            flickr8k_path="data/flickr8k",
            cifar10_path="data/cifar10",
            processor=self.processor,
            split="train",
            max_samples=20000
        )
        self.assertEqual(dataset.total_size, min(2 + 1000, 20000))

    @patch('utils.data_loader.Image.open')
    @patch('utils.data_loader.pd.read_csv')
    @patch('utils.data_loader.CIFAR10')
    def test_getitem_flickr8k(self, mock_cifar, mock_csv, mock_image):
        mock_csv.return_value = pd.DataFrame({
            "image": ["img1.jpg"],
            "caption": ["Caption 1"]
        })
        mock_cifar.return_value = MagicMock(__len__=lambda x: 1000)
        mock_image.return_value.convert.return_value = Image.new('RGB', (224, 224))
        self.processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (1, 128)),
            "attention_mask": torch.ones(1, 128)
        }
        dataset = MultimodalDataset(
            flickr8k_path="data/flickr8k",
            cifar10_path="data/cifar10",
            processor=self.processor,
            split="train"
        )
        item = dataset[0]
        self.assertIn("image", item)
        self.assertIn("text", item)
        self.assertIn("attention_mask", item)

    @patch('utils.data_loader.get_dataloaders')
    def test_get_dataloaders(self, mock_dataloaders):
        train_loader = MagicMock()
        val_loader = MagicMock()
        mock_dataloaders.return_value = (train_loader, val_loader)
        train_dl, val_dl = get_dataloaders(self.config, self.processor)
        self.assertEqual(train_dl, train_loader)
        self.assertEqual(val_dl, val_loader)

if __name__ == '__main__':
    unittest.main()