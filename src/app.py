from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import numpy as np
import tensorflow as tf
from YingYanAI import YingYanAI
from logger_config import setup_logger

# 设置日志记录器
logger = setup_logger(__name__)

app = FastAPI(
    title="鹰眼AI API",
    description="智能图像识别服务API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_names = []
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@app.on_event("startup")
def load_model():
    """加载模型和类别名称"""
    global model, class_names

    model_path = "../models/yingyan_model.h5"
    class_names_path = "models/class_names.txt"

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型文件不存在，请先运行 python src/YingYanAI.py 训练模型",
        )

    try:
        model = tf.keras.models.load_model(model_path)
        if os.path.exists(class_names_path):
            with open(class_names_path, "r") as f:
                class_names = f.read().splitlines()
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型加载失败，请确保模型文件完整",
        )


# Load model at startup
load_model()


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(file: UploadFile = File(...)):
    """图像预测接口"""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="模型未加载，请先运行 python src/YingYanAI.py 训练模型",
        )

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
