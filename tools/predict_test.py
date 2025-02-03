# 导入所需库
import requests
import sys
from pathlib import Path
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.logger_config import setup_logger
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"当前Python路径: {sys.path}")
    sys.exit(1)

# 设置日志记录器
logger = setup_logger(__name__)


def test_prediction():
    url = "http://localhost:8000/predict"

    try:
        # 获取图片路径
        image_path = (
            input("[鹰眼AI] 请拖拽图片到此处: ")
            .strip()
            .replace("'", "")
            .replace('"', "")
        )

        # 验证文件是否存在
        if not Path(image_path).is_file():
            logger.error("找不到图片文件！")
            return

        # 验证文件类型
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        if not Path(image_path).suffix.lower() in valid_extensions:
            logger.error("不支持的图片格式！仅支持 jpg、jpeg、png、bmp 格式")
            return

        logger.info(f"正在处理图片: {Path(image_path).name}")

        # 发送请求
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        # 处理响应
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                logger.info("预测结果:")
                logger.info(f"类别: {result['predicted_class']}")
                logger.info(f"置信度: {result['confidence']:.2%}")
            else:
                logger.error(f"预测失败: {result.get('error', '未知错误')}")
        else:
            logger.error(f"请求失败: HTTP {response.status_code}")

    except KeyboardInterrupt:
        logger.warning("已取消操作")
    except Exception as e:
        logger.error(f"发生错误: {str(e)}")


if __name__ == "__main__":
    logger.info("鹰眼AI图像预测测试工具启动")
    logger.info("=" * 30)
    test_prediction()
