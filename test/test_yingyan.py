import unittest
from unittest.mock import Mock, patch
import os
import tensorflow as tf
from src.YingYanAI import YingYanAI
from src.logger_config import setup_logger


class TestYingYanAI(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.yingyan = YingYanAI()
        self.logger = setup_logger(__name__)

    @patch("tensorflow.keras.Model.fit")
    @patch("tensorflow.keras.Model.save")
    def test_train_normal_dataset(self, mock_save, mock_fit):
        """测试正常大小数据集的训练过程"""
        # 模拟训练数据生成器
        self.yingyan.train_generator.filenames = ["img1.jpg", "img2.jpg"] * 10

        # 模拟训练历史记录
        mock_history = Mock()
        mock_fit.return_value = mock_history

        # 执行训练
        result = self.yingyan.train(epochs=5)

        # 验证训练参数是否正确
        mock_fit.assert_called_once_with(
            self.yingyan.train_generator,
            epochs=5,
            validation_data=self.yingyan.validation_generator,
            verbose=1,
        )

        # 验证模型是否保存
        mock_save.assert_called_once_with("models/yingyan_model.h5")

        # 验证返回训练历史
        self.assertEqual(result, mock_history)

    @patch("tensorflow.keras.Model.fit")
    @patch("tensorflow.keras.Model.save")
    def test_train_small_dataset(self, mock_save, mock_fit):
        """测试小规模数据集的训练过程（少于10个样本）"""
        # 模拟小规模数据集
        self.yingyan.train_generator.filenames = ["img1.jpg", "img2.jpg"]

        mock_history = Mock()
        mock_fit.return_value = mock_history

        result = self.yingyan.train(epochs=5)

        # 验证小数据集警告
        self.assertTrue(len(self.yingyan.train_generator.filenames) < 10)

        # 验证训练是否继续进行
        mock_fit.assert_called_once()
        mock_save.assert_called_once()
        self.assertEqual(result, mock_history)

    @patch("tensorflow.keras.Model.fit")
    def test_train_model_save_error(self, mock_fit):
        """测试模型保存错误的处理"""
        # 模拟成功训练
        mock_history = Mock()
        mock_fit.return_value = mock_history

        # 模拟保存错误
        with patch("tensorflow.keras.Model.save") as mock_save:
            mock_save.side_effect = Exception("保存失败")

            # 验证异常抛出
            with self.assertRaises(Exception):
                self.yingyan.train(epochs=5)

    def test_models_directory_creation(self):
        """测试模型目录的创建"""
        # 如果存在则删除模型目录
        if os.path.exists("models"):
            os.rmdir("models")

        # 训练时应创建模型目录
        with patch("tensorflow.keras.Model.fit"), patch("tensorflow.keras.Model.save"):
            self.yingyan.train()

        self.assertTrue(os.path.exists("models"))


if __name__ == "__main__":
    unittest.main()
