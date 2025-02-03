import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from logger_config import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)


class YingYanAI:
    def __init__(self) -> None:
        logger.info(f"TensorFlow 版本: {tf.__version__}")

        # 初始化图像数据生成器，用于数据预处理
        self.train_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            nput_shape=(224, 224, 3),
        )

        # 冻结预训练层
        self.base_model.trainable = False

        # 配置训练数据生成器
        self.train_generator = self.train_datagen.flow_from_directory(
            "images/train",                 # 训练数据目录
            target_size=(224, 224),         # 目标图像尺寸
            batch_size=32,                  # 批次大小
            class_mode="categorical",       # 分类模式：多分类
        )

        # 添加自定义分类层
        self.model = tf.keras.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(2, activation="softmax"),  # 二分类输出层
            ]
        )

    def run(self) -> None:
        logger.info("运行鹰眼AI系统....")

    def train(self) -> None:
        """训练模型"""
        pass


if __name__ == "__main__":
    # 创建并运行鹰眼AI实例
    yyi: YingYanAI = YingYanAI()
    yyi.run()
