from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import os
import numpy as np
import tensorflow as tf
from src.YingYanAI import YingYanAI
from src.logger_config import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

app = FastAPI(title="鹰眼AI API", description="图像识别服务API")

# 加载模型
# Add global variable for class names
model = None
class_names = []


@app.on_event("startup")
async def load_model():
    """启动时加载模型，如果模型不存在则训练新模型"""
    global model, class_names

    model_path = "models/yingyan_model.h5"
    class_names_path = "models/class_names.txt"

    try:
        model = tf.keras.models.load_model(model_path)
        # Load class names if exists
        if os.path.exists(class_names_path):
            with open(class_names_path, "r") as f:
                class_names = f.read().splitlines()
        logger.info("模型加载成功")
    except Exception as e:
        logger.warning(f"模型加载失败: {str(e)}")
        logger.info("开始训练新模型...")

        # 训练新模型
        yyi = YingYanAI()

        history = yyi.train(epochs=10)

        class_names = list(yyi.train_generator.class_indices.keys())
        os.makedirs(os.path.dirname(class_names_path), exist_ok=True)
        with open(class_names_path, "w") as f:
            f.write("\n".join(class_names))

        # 重新加载保存的模型
        model = tf.keras.models.load_model(model_path)
        logger.info("新模型训练完成并加载成功")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 读取和预处理图像
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Use global class_names
        global class_names

        # 进行预测
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        # 获取类别名称
        class_name = (
            class_names[predicted_class] if class_names else str(predicted_class)
        )

        logger.info(
            f"预测完成: 类别 {class_name}({predicted_class}), 置信度 {confidence:.2f}"
        )

        return {
            "success": True,
            "predicted_class": int(predicted_class),
            "class_name": class_name,
            "confidence": confidence,
            "details": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": file.filename,
                "all_probabilities": {
                    class_names[i]: float(predictions[0][i])
                    for i in range(len(class_names))
                },
            },
        }

    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "details": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": getattr(file, "filename", None),
            },
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
