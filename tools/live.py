import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
from pathlib import Path

model_path = "../models/yingyan_model.h5"
model = tf.keras.models.load_model(model_path)


def setup_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("无法打开摄像头，请检查连接")
    return cap


def setup_video_writer(cap):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 100

    # Create videos directory if not exists
    os.makedirs("videos", exist_ok=True)

    # Create video file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"videos/capture_{timestamp}.mp4"

    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def preprocess_image(frame, img_size=(224, 224)):
    # 调整图像尺寸
    frame = cv2.resize(frame, img_size)
    # 归一化处理
    frame = frame / 255.0
    # 添加批次维度
    frame = np.expand_dims(frame, axis=0)
    return frame


def process_frame(frame):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    dimensions = f"Size: {frame.shape[1]}x{frame.shape[0]}"
    cv2.putText(
        frame, dimensions, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # 图像预处理
    processed_frame = preprocess_image(frame)

    # 进行预测
    predictions = model.predict(processed_frame)
    predicted_class = np.argmax(predictions[0])

    # 根据预测结果映射到具体文字信息
    if predicted_class == 1 or predicted_class == 5:
        class_name = "Nude"
        text_color = (0, 0, 255)  # 红色
    elif predicted_class == 3:
        class_name = "No Nude"
        text_color = (0, 255, 0)  # 绿色
    else:
        class_name = f"Unknown class: {predicted_class}"
        text_color = (0, 255, 0)  # 绿色

    # 在图像上显示预测结果
    cv2.putText(
        frame,
        f"Class: {class_name}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        text_color,
        2,
    )

    return frame


def main():
    try:
        cap = setup_camera()
        out = setup_video_writer(cap)

        print("摄像头已启动。按 'q' 退出，'s' 截图")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取画面")
                break

            # Process frame
            frame = process_frame(frame)

            # Display frame
            cv2.imshow("鹰眼AI实时监控", frame)

            # Write frame to video
            out.write(frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("正在退出...")
                break
            elif key == ord("s"):

                os.makedirs("screenshots", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshots/screenshot_{timestamp}.jpg", frame)

                print(f"截图已保存")

    except Exception as e:
        print(f"发生错误: {str(e)}")

    finally:
        if "cap" in locals():
            cap.release()
        if "out" in locals():
            out.release()
        cv2.destroyAllWindows()

        print("已关闭摄像头")


if __name__ == "__main__":
    main()
