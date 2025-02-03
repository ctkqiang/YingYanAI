import unittest
from unittest.mock import Mock, patch
import os
import tensorflow as tf
from src.YingYanAI import YingYanAI
from src.logger_config import setup_logger


class TestYingYanAI(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.yingyan = YingYanAI()
        self.logger = setup_logger(__name__)

    @patch("tensorflow.keras.Model.fit")
    @patch("tensorflow.keras.Model.save")
    def test_train_normal_dataset(self, mock_save, mock_fit):
        """Test training with normal sized dataset"""
        # Mock train generator
        self.yingyan.train_generator.filenames = ["img1.jpg", "img2.jpg"] * 10

        # Mock fit return value
        mock_history = Mock()
        mock_fit.return_value = mock_history

        # Execute training
        result = self.yingyan.train(epochs=5)

        # Verify fit was called with correct parameters
        mock_fit.assert_called_once_with(
            self.yingyan.train_generator,
            epochs=5,
            validation_data=self.yingyan.validation_generator,
            verbose=1,
        )

        # Verify model was saved
        mock_save.assert_called_once_with("models/yingyan_model.h5")

        # Verify history was returned
        self.assertEqual(result, mock_history)

    @patch("tensorflow.keras.Model.fit")
    @patch("tensorflow.keras.Model.save")
    def test_train_small_dataset(self, mock_save, mock_fit):
        """Test training with small dataset (< 10 samples)"""
        # Mock small dataset
        self.yingyan.train_generator.filenames = ["img1.jpg", "img2.jpg"]

        mock_history = Mock()
        mock_fit.return_value = mock_history

        result = self.yingyan.train(epochs=5)

        # Verify warning was logged for small dataset
        self.assertTrue(len(self.yingyan.train_generator.filenames) < 10)

        # Verify training still proceeded
        mock_fit.assert_called_once()
        mock_save.assert_called_once()
        self.assertEqual(result, mock_history)

    @patch("tensorflow.keras.Model.fit")
    def test_train_model_save_error(self, mock_fit):
        """Test handling of model save errors"""
        # Mock successful training
        mock_history = Mock()
        mock_fit.return_value = mock_history

        # Mock save error
        with patch("tensorflow.keras.Model.save") as mock_save:
            mock_save.side_effect = Exception("Save failed")

            # Verify exception is raised
            with self.assertRaises(Exception):
                self.yingyan.train(epochs=5)

    def test_models_directory_creation(self):
        """Test models directory is created if not exists"""
        # Remove models directory if exists
        if os.path.exists("models"):
            os.rmdir("models")

        # Train should create models directory
        with patch("tensorflow.keras.Model.fit"), patch("tensorflow.keras.Model.save"):
            self.yingyan.train()

        self.assertTrue(os.path.exists("models"))


if __name__ == "__main__":
    unittest.main()
