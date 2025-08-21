# YOLO+ResNet50 融合模型训练指南

## 概述

本指南将帮助您训练一个基于Food20_new数据集的YOLO+ResNet50融合模型，用于食物识别和尿酸含量分析。

## 数据集准备

### 1. 数据集结构

确保您的数据集结构如下：
```
datasets/Food20_new/
├── train/
│   ├── images/
│   │   ├── food1.jpg
│   │   ├── food2.jpg
│   │   └── ...
│   └── train.json
├── val/
│   ├── images/
│   │   ├── food1.jpg
│   │   ├── food2.jpg
│   │   └── ...
│   └── val.json
└── test/
    ├── images/
    │   ├── food1.jpg
    │   ├── food2.jpg
    │   └── ...
    └── test.json
```

### 2. 数据集检查

运行以下命令检查数据集：
```bash
python start_training.py
```

## 训练配置

### 默认配置

- **数据集**: Food20_new
- **模型**: YOLO+ResNet50融合
- **批次大小**: 16 (可根据GPU内存调整)
- **训练轮数**: 30
- **学习率**: 0.001
- **优化器**: AdamW
- **数据增强**: 随机翻转、旋转、颜色抖动

### 自定义配置

您可以通过命令行参数自定义训练配置：

```bash
python train_yolo_resnet.py \
    --data_dir datasets/Food20_new \
    --output_dir outputs/training \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.0005 \
    --num_workers 4
```

## 训练步骤

### 1. 快速开始

使用Windows批处理文件：
```bash
start_training.bat
```

或使用Python脚本：
```bash
python start_training.py
```

### 2. 手动训练

```bash
python train_yolo_resnet.py
```

### 3. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `datasets/Food20_new` | 数据集目录 |
| `--output_dir` | `outputs/training` | 输出目录 |
| `--batch_size` | `16` | 批次大小 |
| `--epochs` | `30` | 训练轮数 |
| `--learning_rate` | `0.001` | 学习率 |
| `--weight_decay` | `1e-4` | 权重衰减 |
| `--lr_step_size` | `20` | 学习率调度步长 |
| `--lr_gamma` | `0.1` | 学习率调度衰减因子 |
| `--num_workers` | `2` | 数据加载器工作进程数 |
| `--save_interval` | `10` | 模型保存间隔 |

## 训练输出

### 1. 输出目录结构

```
outputs/training/
├── logs/
│   └── training.log
├── models/
│   ├── best_model.pth
│   ├── final_model.pth
│   └── checkpoint_epoch_*.pth
└── plots/
    ├── training_curves.png
    └── confusion_matrix.png
```

### 2. 模型文件

- **best_model.pth**: 验证集上表现最好的模型
- **final_model.pth**: 训练完成后的最终模型
- **checkpoint_epoch_*.pth**: 定期保存的检查点

### 3. 训练日志

训练过程中会生成详细的日志文件，包括：
- 训练进度
- 损失和准确率变化
- 模型性能指标
- 错误信息

### 4. 可视化结果

- **训练曲线**: 显示训练和验证损失/准确率变化
- **混淆矩阵**: 显示各类别的分类性能

## 模型集成

### 1. 自动集成

训练完成后，后端服务会自动检测并使用训练好的模型：

```python
# 后端会自动加载最佳模型
trained_model = load_trained_model()
```

### 2. 手动指定模型

```python
from models.trained_model import TrainedYOLOResNetModel

# 指定模型路径
model = TrainedYOLOResNetModel('outputs/training/models/best_model.pth')
```

### 3. 模型切换

后端服务支持自动切换：
- 如果存在训练好的模型，优先使用训练好的模型
- 如果没有训练好的模型，使用预训练模型

## 性能优化

### 1. GPU加速

确保安装了CUDA版本的PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 内存优化

- 减小批次大小
- 减少数据加载器工作进程数
- 使用梯度累积

### 3. 训练加速

- 使用混合精度训练
- 启用数据预取
- 使用更快的存储设备

## 故障排除

### 1. 常见问题

**问题**: CUDA内存不足
**解决**: 减小批次大小或使用CPU训练

**问题**: 数据集加载失败
**解决**: 检查数据集路径和格式

**问题**: 模型保存失败
**解决**: 确保输出目录有写权限

### 2. 日志分析

查看训练日志文件：
```bash
tail -f outputs/training/logs/training.log
```

### 3. 性能监控

训练过程中监控：
- GPU使用率
- 内存使用情况
- 训练速度

## 模型评估

### 1. 自动评估

训练完成后会自动生成：
- 分类报告 (classification_report.json)
- 混淆矩阵图
- 训练曲线图

### 2. 手动评估

```python
from models.trained_model import load_trained_model

# 加载模型
model = load_trained_model()

# 测试图像
test_image = Image.open('test_image.jpg')
result = model.predict(test_image)
print(result)
```

## 部署建议

### 1. 模型选择

- **生产环境**: 使用 `best_model.pth`
- **开发测试**: 使用 `final_model.pth`

### 2. 性能要求

- **CPU**: 推荐4核以上
- **内存**: 推荐8GB以上
- **存储**: 推荐SSD

### 3. 监控

- 模型推理时间
- 准确率变化
- 错误率统计

## 更新和维护

### 1. 模型更新

定期重新训练模型以保持性能：
```bash
# 使用新数据重新训练
python train_yolo_resnet.py --epochs 50
```

### 2. 版本管理

- 保存不同版本的模型
- 记录训练配置和结果
- 建立模型性能基准

### 3. 自动化

- 设置定时训练任务
- 自动模型评估
- 自动部署最佳模型

## 联系支持

如果在训练过程中遇到问题，请：
1. 查看训练日志
2. 检查数据集格式
3. 验证环境配置
4. 参考故障排除部分

---

**注意**: 训练时间取决于硬件配置和数据集大小，通常需要1-4小时。 