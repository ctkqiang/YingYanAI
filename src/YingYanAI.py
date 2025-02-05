import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from logger_config import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)


class YingYanAI:
    def __init__(
        self, train_dir="images/train", img_size=(224, 224), batch_size=32
    ) -> None:
        """
        初始化鹰眼AI系统
        参数说明：
            train_dir: 训练数据集目录路径
            img_size: 图像处理的目标尺寸，MobileNetV2要求输入图像尺寸为(224, 224)
            batch_size: 批处理大小，影响训练速度和内存使用
        """
        logger.info(f"TensorFlow 版本: {tf.__version__}")

        # 配置训练数据增强器，通过对图像进行变换来增加训练样本的多样性，防止过拟合
        self.train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,  # 图像归一化，将像素值缩放到0-1范围
            rotation_range=20,  # 随机旋转图像，角度范围为±20度
            width_shift_range=0.2,  # 随机水平平移图像，范围为宽度的±20%
            height_shift_range=0.2,  # 随机垂直平移图像，范围为高度的±20%
            horizontal_flip=True,  # 随机水平翻转图像
            zoom_range=0.2,  # 随机缩放图像，范围为原尺寸的80%-120%
        )

        # 验证数据只需要归一化处理，不需要数据增强
        self.val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        # 加载预训练的MobileNetV2模型，移除顶层分类器
        self.base_model = MobileNetV2(
            weights="imagenet",  # 使用在ImageNet数据集上预训练的权重
            include_top=False,  # 不包含顶层分类器
            input_shape=(img_size[0], img_size[1], 3),  # 设置输入图像尺寸和通道数
        )

        self.base_model.trainable = False  # 冻结预训练模型的权重，只训练新添加的分类层

        # 设置训练数据生成器，从目录中加载和预处理训练图像
        self.train_generator = self.train_datagen.flow_from_directory(
            train_dir,  # 训练数据目录
            target_size=img_size,  # 调整图像尺寸
            batch_size=batch_size,  # 每批处理的图像数量
            class_mode="categorical",  # 多分类问题的标签编码方式
        )

        # 设置验证数据生成器，用于评估模型性能
        self.validation_generator = self.val_datagen.flow_from_directory(
            "images/validation",  # 验证数据目录
            target_size=img_size,
            batch_size=batch_size,
            class_mode="categorical",
        )

        # 构建完整的神经网络模型
        self.model = tf.keras.Sequential(
            [
                self.base_model,  # 使用预训练的MobileNetV2作为特征提取器
                tf.keras.layers.GlobalAveragePooling2D(),  # 将特征图转换为向量
                tf.keras.layers.Dropout(0.5),  # 添加Dropout层减少过拟合
                tf.keras.layers.Dense(  # 添加全连接分类层
                    self.train_generator.num_classes,  # 输出类别数量
                    activation="softmax",  # 使用softmax激活函数进行多分类
                    kernel_regularizer=tf.keras.regularizers.l2(
                        0.01
                    ),  # L2正则化防止过拟合
                ),
            ]
        )

        # 配置模型训练参数
        self.model.compile(
            optimizer="adam",  # 使用Adam优化器
            loss="categorical_crossentropy",  # 多分类的损失函数
            metrics=["accuracy"],  # 使用准确率作为评估指标
        )

    def train(self, epochs=10) -> None:
        """训练模型方法"""
        logger.info("开始训练模型...")

        # 检查数据集大小
        num_samples = len(self.train_generator.filenames)
        if num_samples < 10:
            logger.warning("数据集太小，可能导致过拟合！")

        # 开始训练
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,  # 使用验证数据生成器
            verbose=1,
        )

        logger.info("模型训练完成!")

        # 保存模型
        model_path = "../models/yingyan_model.h5"
        os.makedirs("models", exist_ok=True)  # 确保models目录存在
        self.model.save(model_path)
        logger.info(f"模型已保存至: {model_path}")

        return history


if __name__ == "__main__":
    # 创建并运行鹰眼AI实例
    yyi: YingYanAI = YingYanAI()

    # 开始训练
    # 移除 validation_split 参数
    history = yyi.train(epochs=10)
