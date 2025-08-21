# Food20数据集模型训练指南

本指南介绍如何使用Food20_new数据集训练YOLO和ResNet50模型。

## 数据集结构

Food20_new数据集包含以下结构：
```
datasets/Food20_new/
├── train/
│   ├── images/          # 训练图像
│   └── train.json       # 训练标注文件
├── val/
│   ├── images/          # 验证图像
│   └── val.json         # 验证标注文件
└── test/
    ├── images/          # 测试图像
    └── test.json        # 测试标注文件
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于GPU训练)
- 其他依赖见requirements.txt

## 快速开始

### 方法1: 使用批处理脚本 (Windows)

```bash
# 双击运行或在命令行执行
train_food20.bat
```

### 方法2: 使用Shell脚本 (Linux/Mac)

```bash
# 给脚本执行权限
chmod +x train_food20.sh

# 运行脚本
./train_food20.sh
```

### 方法3: 手动执行

```bash
# 1. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics pyyaml pillow numpy opencv-python

# 2. 训练ResNet分类器
python scripts/train_food20_models.py --model resnet

# 3. 训练YOLO检测器
python scripts/train_food20_models.py --model yolo

# 4. 评估模型
python scripts/train_food20_models.py --evaluate
```

## 训练选项

### 训练特定模型

```bash
# 只训练ResNet分类器
python scripts/train_food20_models.py --model resnet

# 只训练YOLO检测器
python scripts/train_food20_models.py --model yolo

# 训练两个模型
python scripts/train_food20_models.py --model both
```

### 使用自定义配置

```bash
# 使用自定义配置文件
python scripts/train_food20_models.py --config my_config.yaml
```

### 只准备数据集

```bash
# 只准备YOLO格式的数据集，不训练
python scripts/train_food20_models.py --prepare-only
```

### 评估模型

```bash
# 评估已训练的模型
python scripts/train_food20_models.py --evaluate
```

## 配置文件

训练配置在 `config/food20_training_config.yaml` 中定义：

### ResNet配置
```yaml
resnet:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  num_classes: 20
  model_save_path: "models/resnet_food20_classifier.pth"
```

### YOLO配置
```yaml
yolo:
  epochs: 100
  batch_size: 16
  learning_rate: 0.01
  model_save_path: "models/yolo_food20_detector.pt"
  image_size: 640
```

## 输出文件

训练完成后，以下文件将被生成：

### 模型文件
- `models/resnet_food20_classifier.pth` - ResNet分类器模型
- `models/yolo_food20_detector.pt` - YOLO检测器模型

### 日志文件
- `logs/` - 训练日志和TensorBoard文件
- `logs/confusion_matrix.png` - 混淆矩阵图

### YOLO数据集
- `datasets/food20_yolo/` - 转换后的YOLO格式数据集

## 模型使用

### 使用ResNet分类器

```python
from models.resnet_classifier import ResNetClassifier

# 加载模型
classifier = ResNetClassifier("models/resnet_food20_classifier.pth")

# 分类图像
result = classifier.classify(image_data)
print(f"预测结果: {result['food_name']}")
```

### 使用YOLO检测器

```python
from models.yolo_detector import YOLODetector

# 加载模型
detector = YOLODetector("models/yolo_food20_detector.pt")

# 检测图像
detections = detector.detect(image_data)
for detection in detections:
    print(f"检测到: {detection['class_name']}")
```

## 性能优化

### GPU训练
- 确保安装了CUDA版本的PyTorch
- 使用 `nvidia-smi` 检查GPU状态
- 调整batch_size以适应GPU内存

### 内存优化
- 减少batch_size
- 使用梯度累积
- 启用混合精度训练

### 数据加载优化
- 增加num_workers
- 使用pin_memory=True
- 使用SSD存储数据集

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 使用更小的模型
   - 启用梯度检查点

2. **数据集加载错误**
   - 检查数据集路径
   - 验证JSON文件格式
   - 确保图像文件存在

3. **训练不收敛**
   - 调整学习率
   - 增加数据增强
   - 检查损失函数

### 日志查看

```bash
# 查看训练日志
tail -f logs/training.log

# 启动TensorBoard
tensorboard --logdir logs/tensorboard
```

## 扩展功能

### 自定义数据增强
在配置文件中修改数据增强参数：

```yaml
resnet:
  data_augmentation:
    random_horizontal_flip: true
    random_rotation: 15
    color_jitter:
      brightness: 0.3
      contrast: 0.3
```

### 自定义模型架构
修改 `models/resnet_classifier.py` 来使用不同的backbone：

```python
# 使用ResNet101
self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
```

### 多GPU训练
在配置文件中启用多GPU训练：

```yaml
training:
  device: "cuda"
  multi_gpu: true
  gpu_ids: [0, 1]
```

## 许可证

本项目遵循MIT许可证。Food20数据集遵循Apache License 2.0。 