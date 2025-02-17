# 🦅 鹰眼AI

> 像鹰一样锐利，如AI般智能

一款革命性的计算机视觉系统，采用尖端人工智能技术，将图像分析提升至新高度。基于谷歌高效轻量级 MobileNetV2 架构，实现快速精准的视觉智能化。

## 🌟 项目亮点

鹰眼AI是新一代智能视觉平台，融合深度学习与迁移学习技术，打造高性能视觉分析引擎。系统继承 MobileNetV2 预训练模型的优秀特性，实现快速部署、高效推理、精准识别的完美统一。

## 💫 核心优势

- 🎯 智能图像分类 - 95%+ 的分类准确率
- 🚀 高性能API服务 - 毫秒级响应速度
- 🤖 自动化模型训练 - 一键启动迭代优化
- ⚡️ 实时预测能力 - 支持流式数据处理
- 📥 智能数据采集 - 自动构建训练数据集

## 🛠 技术栈

- 🧠 TensorFlow 2.x - 强大的深度学习引擎
- ⚡️ FastAPI - 高性能异步Web框架
- 📱 MobileNetV2 - 轻量级神经网络架构
- 🐍 Python 3.6+ - 稳定可靠的开发语言

## 项目结构

```bash
YiShouAI/
├── src/                         # 源代码目录
│   ├── YingYanAI.py             # 核心AI模型实现
│   ├── app.py                   # FastAPI 服务实现
│   ├── download_data.py         # 数据采集工具
│   └── logger_config.py          # 日志配置
├── tools/                       # 实用工具目录
│   └── predict_test.py          # 预测功能测试工具
├── models/                      # 模型存储目录
│   ├── yingyan_model.h5         # 训练后的模型文件
│   └── class_names.txt          # 类别名称映射文件
├── images/                      # 图像数据目录
│   ├── train/                   # 训练数据集
│   │   ├── {关键词1}/            # 根据搜索关键词自动创建的类别目录
│   │   └── {关键词2}/            # 根据搜索关键词自动创建的类别目录
│   ├── validation/              # 验证数据集
│   └── test/                    # 测试数据集
├── log/                         # 日志文件目录
│   └── *.log                    # 运行日志文件
└── README.md                    # 项目说明文档
```


## 快速开始
### 环境要求

```bash
# 安装依赖
pip install tensorflow fastapi uvicorn python-multipart pillow icrawler

# 创建必要的目录
mkdir -p images/train images/validation models log
```


### 数据采集与准备
#### 使用数据采集工具：

创建必要的数据目录：
```bash
mkdir -p images/{train,validation,test}
```

使用数据采集工具：
```bash
python src/download_data.py
```

关键词格式说明：

- 使用下划线连接多个单词
- 示例关键词格式：

```plaintext
  elon_musk          # 人物类别
  cute_cat           # 动物类别
  red_apple          # 物品类别
  sport_car          # 交通工具
  modern_building    # 建筑类别
  m*a_kh*lifa        # 模糊匹配
  mark_zuckerberg    # 人物类别
```

示例操作：
```bash
请输入搜索关键词: porn_nude # 限制级内容
# 或
请输入搜索关键词: apple     # 普通内容
是否为训练集？(y/n): y
需要下载多少张图片？(默认: 5): 10
```

注意：请确保在 images 目录下创建以下子目录：

- train/ : 训练数据集
- validation/ : 验证数据集
- test/ : 测试数据集

#### 直接训练模型

完成数据采集后，可以直接运行模型训练：

```bash
python src/YingYanAI.py
```

训练过程输出示例：
```bash
[鹰眼 AI]: 2025-02-04 12:44:24,490 - **main** - INFO - TensorFlow 版本: 2.13.0
Found 15 images belonging to 3 classes.
Found 15 images belonging to 3 classes.
[鹰眼 AI]: 2025-02-04 12:44:26,852 - **main** - INFO - 开始训练模型...
Epoch 1/10
1/1 [==============================] - 6s 6s/step - loss: 1.2042 - accuracy: 0.4000 - val_loss: 1.2801 - val_accuracy: 0.4667
Epoch 2/10
1/1 [==============================] - 2s 2s/step - loss: 1.3403 - accuracy: 0.4000 - val_loss: 1.1074 - val_accuracy: 0.5333
Epoch 3/10
1/1 [==============================] - 1s 1s/step - loss: 1.6560 - accuracy: 0.2667 - val_loss: 0.9599 - val_accuracy: 0.5333
Epoch 4/10
1/1 [==============================] - 2s 2s/step - loss: 1.5020 - accuracy: 0.4667 - val_loss: 0.8382 - val_accuracy: 0.7333
Epoch 5/10
1/1 [==============================] - 1s 1s/step - loss: 1.0760 - accuracy: 0.4667 - val_loss: 0.7422 - val_accuracy: 0.7333
Epoch 6/10
1/1 [==============================] - 1s 1s/step - loss: 1.0561 - accuracy: 0.6000 - val_loss: 0.6698 - val_accuracy: 0.8667
Epoch 7/10
1/1 [==============================] - 1s 1s/step - loss: 1.0602 - accuracy: 0.3333 - val_loss: 0.6113 - val_accuracy: 0.8667
Epoch 8/10
1/1 [==============================] - 1s 1s/step - loss: 0.6969 - accuracy: 0.8000 - val_loss: 0.5660 - val_accuracy: 0.8667
Epoch 9/10
1/1 [==============================] - 1s 1s/step - loss: 0.8998 - accuracy: 0.6000 - val_loss: 0.5307 - val_accuracy: 0.8667
Epoch 10/10
1/1 [==============================] - 2s 2s/step - loss: 0.9569 - accuracy: 0.7333 - val_loss: 0.5009 - val_accuracy: 0.8667
[鹰眼 AI]: 2025-02-04 12:44:47,790 - **main** - INFO - 模型训练完成!
[鹰眼 AI]: 2025-02-04 12:44:48,118 - **main** - INFO - 模型已保存至: models/yingyan_model.h5
```
训练过程监控:
- 每个 epoch 会显示:
  - loss: 训练损失值
  - accuracy: 训练准确率
  - val_loss: 验证损失值
  - val_accuracy: 验证准确率
- 训练完成后模型自动保存到 models/yingyan_model.h5
- 详细日志记录在 log 目录下

## 🖥 控制台应用程序

### 启动控制台

```bash
python run.py
```

控制台应用程序提供以下功能：

1. 启动 API 服务

   - 启动 FastAPI 服务器
   - 默认端口：8000
   - 支持热重载

2. 启动实时监控

   - 打开摄像头进行实时内容识别
   - 支持视频录制和截图
   - 实时显示识别结果

3. 训练模型

   - 一键启动模型训练
   - 自动保存训练结果
   - 显示训练进度

4. 系统状态

   - 查看 AI 模型加载状态
   - 检查 API 服务就绪情况
   - 监控实时监控系统状态

### 操作说明

- 使用数字键 1-4 选择对应功能
- 按 q 退出程序
- 按 Enter 确认选择
- 系统状态界面按任意键返回主菜单

### 界面预览
```bash
╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ │
│ 鹰眼 AI 系统控制台 │
│ ================= │
│ │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────┬──────────────╮
│ 选项 │ 描述          │
├──────┼──────────────┤
│ 1 │ 启动 API 服务    │
│ 2 │ 启动实时监控      │
│ 3 │ 训练模型         │
│ 4 │ 系统状态         │
│ q │ 退出系统         │
╰──────┴──────────────╯
请选择操作 [1/2/3/4/q]: 
```

### 启动服务

1. 开发环境
```bash
cd src && uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

2. 生产环境
```bash
cd src && uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 4
```

服务将在 http://localhost:8000 启动，可通过 http://localhost:8000/docs 访问交互式 API 文档。

## API 使用说明

### 预测接口

- 端点：`POST /predict`
- 功能：上传图片进行分类预测
- 示例：

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
```

### 预测测试工具

位于 tools/predict_test.py ，用于快速测试模型的预测功能。

使用方法：
> 确保 API 服务已启动

```bash
python tools/predict_test.py
```

输出示例：

```bash
[鹰眼 AI]: 2025-02-04 03:46:15,189 - **main** - INFO - 鹰眼 AI 图像预测测试工具启动
[鹰眼 AI]: 2025-02-04 03:46:15,189 - **main** - INFO - ==============================
[鹰眼 AI] 请拖拽图片到此处: '/path/to/images/train/porn_nude/000001.jpg'
[鹰眼 AI]: 2025-02-04 03:46:19,696 - **main** - INFO - 正在处理图片: 000001.jpg
[鹰眼 AI]: 2025-02-04 03:46:20,477 - **main** - INFO - 预测结果:
[鹰眼 AI]: 2025-02-04 03:46:20,477 - **main** - INFO - 类别: 1
[鹰眼 AI]: 2025-02-04 03:46:20,477 - **main** - INFO - 置信度: 98.90%
[鹰眼 AI]: 2025-02-04 03:46:20,477 - **main** - INFO - 是否包含限制级内容: 【是】
```


功能特点：

- 支持拖拽图片文件进行测试
- 自动验证图片格式（支持 jpg、jpeg、png、bmp）
- 详细的日志记录
- 友好的中文交互界面
- 显示预测结果和置信度
  
示例输出:
```bash
[鹰眼 AI] 2024-02-04 15:30:45 - INFO - 鹰眼 AI 图像预测测试工具启动
[鹰眼 AI] ==============================
[鹰眼 AI] 请拖拽图片到此处: /path/to/image.jpg
[鹰眼 AI] 正在处理图片: image.jpg
[鹰眼 AI] 预测结果:
[鹰眼 AI] 类别: 1
[鹰眼 AI] 置信度: 95.23%
```

响应格式：

```json
{
  "success": true,
  "predicted_class": 0,
  "confidence": 0.95
}
```

### 实时监控功能

使用实时监控工具可以通过摄像头进行即时内容识别：
```bash
python tools/live.py
````

功能特点：

- 🎥 实时摄像头画面捕捉
- 🔍 即时内容分析与识别
- ⚡️ 毫秒级响应速度
- 📊 实时显示识别结果
- 📸 支持截图功能
- 📹 自动录制视频
  操作说明：

- 按 'q' 键退出程序
- 按 's' 键保存当前画面截图
  显示信息：

- 时间戳
- 画面尺寸
- 识别结果（安全/限制级）
- 识别置信度
  存储位置：

- 视频文件： videos/capture\_[时间戳].mp4
- 截图文件： screenshots/screenshot\_[时间戳].jpg
  注意：首次运行前请确保已完成模型训练，且模型文件存在于 models 目录下。


## 特性

- 自动数据增强
- 迁移学习
- 模型自动训练
- RESTful API
- 实时预测
- 详细的日志记录

## 注意事项

- 确保训练数据集在 `images/train` 目录下按类别分类
- 验证数据集放置在 `images/validation` 目录下
- 首次启动服务时如果没有预训练模型会自动开始训练
- 所有操作日志都会记录在 `log` 目录下

## 系统要求

- CPU 或 GPU (推荐)
- 内存：4GB 以上
- 磁盘空间：500MB 以上

## 算法
![鹰眼AI系统流程图](https://github.com/ctkqiang/YingYanAI/blob/main/assets/algorithm.png?raw=true)

## 实时监控系统算法
![鹰眼AI系统流程图](https://github.com/ctkqiang/YingYanAI/blob/main/assets/live.png?raw=true)



## 评估模型：
在 `src/YingYanAI.py` 中添加以下代码来评估模型性能：

```python
from sklearn.metrics import classification_report

# 加载测试数据

test_generator = val_datagen.flow_from_directory(
  "images/test",
  target_size=img_size,
  batch_size=batch_size,
  class_mode="categorical",
  shuffle=False
)

# 评估模型

predictions = yyi.model.predict(test_generator)
y_pred = tf.argmax(predictions, axis=1).numpy()
y_true = test_generator.classes

print(classification_report(y_true, y_pred))

```

## 预测新图像：
使用保存的模型对新图像进行预测：
```python
from tensorflow.keras.models import load_model

model = load_model("models/yingyan_model.h5")
image_path = "path_to_image.jpg"

img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # 添加批次维度

predictions = model.predict(img_array)

print(f"预测结果: {predictions}")
```

## 参数说明
## 参数说明

| 参数名     | 说明               | 默认值           |
| ---------- | ------------------ | ---------------- |
| train_dir  | 训练数据目录路径   | `"images/train"` |
| img_size   | 图像处理的目标尺寸 | `(224, 224)`     |
| batch_size | 批处理大小         | `32`             |
| epochs     | 训练周期数         | `10`             |

> 注：
>
> - img_size 必须为 (224, 224)，这是 MobileNetV2 的要求
> - batch_size 可根据可用内存大小调整
> - epochs 可根据训练效果调整

## 模型架构
1. MobileNetV2：作为基础特征提取器，使用 ImageNet 预训练权重。
2. 全局平均池化：将特征图压缩为一维向量。
3. Dropout：随机丢弃 50% 神经元，防止过拟合。
4. 全连接分类层：使用 Softmax 激活函数进行多分类。

## 性能评估
#### 评估指标
- 准确率 (Accuracy)：模型预测正确的比例。
- 召回率 (Recall)：真正例中被正确预测的比例。
- F1 分数：综合衡量模型的精度和召回率。


## 模型评估报告

| 类别         | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| 0            | 0.95      | 0.94   | 0.95     | 34      |
| 1            | 0.96      | 0.97   | 0.96     | 36      |
| **准确率**   | -         | -      | 0.95     | 70      |
| **宏平均**   | 0.95      | 0.95   | 0.95     | 70      |
| **加权平均** | 0.95      | 0.95   | 0.95     | 70      |

> 注：
>
> - Precision: 精确率
> - Recall: 召回率
> - F1-Score: F1 分数
> - Support: 样本数量

## 许可证

本项目采用 **木兰宽松许可证 (Mulan PSL)** 进行许可。  
有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

[![License: Mulan PSL v2](https://img.shields.io/badge/License-Mulan%20PSL%202-blue.svg)](http://license.coscl.org.cn/MulanPSL2)


## 🌟 开源项目赞助计划

### 用捐赠助力发展

感谢您使用本项目！您的支持是开源持续发展的核心动力。  
每一份捐赠都将直接用于：  
✅ 服务器与基础设施维护  
✅ 新功能开发与版本迭代  
✅ 文档优化与社区建设

点滴支持皆能汇聚成海，让我们共同打造更强大的开源工具！

---

### 🌐 全球捐赠通道

#### 国内用户

<div align="center" style="margin: 40px 0">

<div align="center">
<table>
<tr>
<td align="center" width="300">
<img src="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9863.jpg?raw=true" width="200" />
<br />
<strong>🔵 支付宝</strong>
</td>
<td align="center" width="300">
<img src="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9859.JPG?raw=true" width="200" />
<br />
<strong>🟢 微信支付</strong>
</td>
</tr>
</table>
</div>
</div>

#### 国际用户

<div align="center" style="margin: 40px 0">
  <a href="https://qr.alipay.com/fkx19369scgxdrkv8mxso92" target="_blank">
    <img src="https://img.shields.io/badge/Alipay-全球支付-00A1E9?style=flat-square&logo=alipay&logoColor=white&labelColor=008CD7">
  </a>
  
  <a href="https://ko-fi.com/F1F5VCZJU" target="_blank">
    <img src="https://img.shields.io/badge/Ko--fi-买杯咖啡-FF5E5B?style=flat-square&logo=ko-fi&logoColor=white">
  </a>
  
  <a href="https://www.paypal.com/paypalme/ctkqiang" target="_blank">
    <img src="https://img.shields.io/badge/PayPal-安全支付-00457C?style=flat-square&logo=paypal&logoColor=white">
  </a>
  
  <a href="https://donate.stripe.com/00gg2nefu6TK1LqeUY" target="_blank">
    <img src="https://img.shields.io/badge/Stripe-企业级支付-626CD9?style=flat-square&logo=stripe&logoColor=white">
  </a>
</div>

---

### 📌 开发者社交图谱

#### 技术交流

<div align="center" style="margin: 20px 0">
  <a href="https://github.com/ctkqiang" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-开源仓库-181717?style=for-the-badge&logo=github">
  </a>
  
  <a href="https://stackoverflow.com/users/10758321/%e9%92%9f%e6%99%ba%e5%bc%ba" target="_blank">
    <img src="https://img.shields.io/badge/Stack_Overflow-技术问答-F58025?style=for-the-badge&logo=stackoverflow">
  </a>
  
  <a href="https://www.linkedin.com/in/ctkqiang/" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-职业网络-0A66C2?style=for-the-badge&logo=linkedin">
  </a>
</div>

#### 社交互动

<div align="center" style="margin: 20px 0">
  <a href="https://www.instagram.com/ctkqiang" target="_blank">
    <img src="https://img.shields.io/badge/Instagram-生活瞬间-E4405F?style=for-the-badge&logo=instagram">
  </a>
  
  <a href="https://twitch.tv/ctkqiang" target="_blank">
    <img src="https://img.shields.io/badge/Twitch-技术直播-9146FF?style=for-the-badge&logo=twitch">
  </a>
  
  <a href="https://github.com/ctkqiang/ctkqiang/blob/main/assets/IMG_9245.JPG?raw=true" target="_blank">
    <img src="https://img.shields.io/badge/微信公众号-钟智强-07C160?style=for-the-badge&logo=wechat">
  </a>
</div>

---

🙌 感谢您成为开源社区的重要一员！  
💬 捐赠后欢迎通过社交平台与我联系，您的名字将出现在项目致谢列表！
