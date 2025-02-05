import tensorflow as tf

model_path = "../models/yingyan_model.h5"
model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 保存为 .tflite 文件
tflite_model_path = "../models/yingyan_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)
