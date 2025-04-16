import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from utils.data_loader import MultimodalDataset, get_dataloaders
from PIL import Image
import numpy as np

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
        self.processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "input_ids": torch.randint(0, 1000, (1, 128)),
            "attention_mask": torch.ones(1, 128)
        }

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

    @patch('utils.data_loader.pd.read_csv')
    @patch('utils.data_loader.CIFAR10')
    def test_getitem_cifar10(self, mock_cifar, mock_csv):
        # Mock Flickr8k
        mock_csv.return_value = pd.DataFrame({
            "image": ["img1.jpg"],
            "caption": ["Caption 1"]
        })

        # 建立 CIFAR-10 mock dataset
        mock_dataset = MagicMock()
        mock_image = Image.fromarray(np.uint8(np.random.rand(32, 32, 3) * 255))
        mock_dataset.__getitem__.side_effect = lambda idx: (mock_image, 1)
        mock_dataset.__len__.return_value = 1
        mock_dataset.classes = ["airplane", "automobile"]
        mock_cifar.return_value = mock_dataset

        dataset = MultimodalDataset(
            flickr8k_path="data/flickr8k",
            cifar10_path="data/cifar10",
            processor=self.processor,
            split="train"
        )
        item = dataset[1]
        self.assertIn("image", item)
        self.assertIn("text", item)
        self.assertIn("attention_mask", item)

    @patch('utils.data_loader.pd.read_csv')
    @patch('utils.data_loader.CIFAR10')
    @patch('utils.data_loader.Image.open')
    
    def test_get_dataloaders(self, mock_image_open, mock_cifar, mock_csv):
        # Mock Flickr8k captions
        mock_csv.return_value = pd.DataFrame({
            "image": ["img1.jpg", "img2.jpg"],
            "caption": ["Caption 1", "Caption 2"]
        })

        # 建立 CIFAR-10 mock dataset
        mock_dataset = MagicMock()
        mock_image = Image.new("RGB", (32, 32))
        mock_dataset.__getitem__.side_effect = lambda idx: (mock_image, 0)
        mock_dataset.__len__.return_value = 100
        mock_dataset.classes = ["cat", "dog"]
        mock_cifar.return_value = mock_dataset

        # Mock image open
        mock_image_open.return_value.convert.return_value = Image.new("RGB", (224, 224))

        train_dl, val_dl = get_dataloaders(self.config, self.processor)
        self.assertTrue(isinstance(train_dl, torch.utils.data.DataLoader))
        self.assertTrue(isinstance(val_dl, torch.utils.data.DataLoader))

        batch = next(iter(train_dl))
        self.assertIn("image", batch)
        self.assertIn("text", batch)
        self.assertIn("attention_mask", batch)
if __name__ == '__main__':
    unittest.main()
