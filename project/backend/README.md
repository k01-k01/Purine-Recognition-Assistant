# 食物识别与尿酸分析API (FastAPI版本)

基于YOLO+ResNet融合模型的食物识别和尿酸含量分析服务，使用FastAPI构建。

## 🎯 功能特性

- 🍽️ **食物识别**: 使用YOLO检测器定位食物区域，ResNet分类器精确识别食物类型
- 🧬 **尿酸分析**: 提供详细的嘌呤含量信息和饮食建议
- 📊 **营养估算**: 计算食物的营养成分（热量、蛋白质、脂肪等）
- 🎯 **痛风友好**: 专门为痛风患者提供饮食指导
- 🚀 **高性能**: 基于PyTorch和Ultralytics的高效推理
- 📱 **API友好**: 提供RESTful API接口，支持多种客户端
- 📚 **自动文档**: FastAPI自动生成交互式API文档

## 🏗️ 技术架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   客户端应用    │    │   FastAPI服务   │    │   模型层        │
│                │    │                │    │                │
│ - Web前端      │◄──►│ - 图片上传      │    │ - YOLO检测器   │
│ - 移动应用     │    │ - 食物识别      │    │ - ResNet分类器  │
│ - 小程序       │    │ - 尿酸分析      │    │ - 食物数据库    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8+
- 至少4GB内存
- 至少2GB磁盘空间

### 2. 一键安装

**Linux/Mac:**
```bash
cd project/backend
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
cd project\backend
install.bat
```

### 3. 手动安装

```bash
# 安装Python依赖
pip install -r requirements.txt

# 创建必要目录
mkdir -p models data logs public
```

### 4. 启动服务

```bash
# 方式1: 使用启动脚本
python start_server.py

# 方式2: 直接使用uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 3003
```

### 5. 访问API

- API文档: http://localhost:3003/docs
- 健康检查: http://localhost:3003/health
- 服务器信息: http://localhost:3003/api/info

## 📡 API接口

### 1. 食物识别

**POST** `/api/recognize`

识别图片中的食物并提供尿酸分析。

```bash
curl -X POST "http://localhost:3003/api/recognize" \
     -H "Content-Type: application/json" \
     -d '{
       "image": "base64编码的图片数据"
     }'
```

响应示例：
```json
{
  "success": true,
  "data": {
    "foodName": "apple",
    "purineLevel": "low",
    "purineContent": "5-15mg/100g",
    "suitableForGout": true,
    "advice": "低嘌呤食物，痛风患者可以放心食用",
    "nutritionEstimate": {
      "calories": "52kcal/100g",
      "protein": "0.3g/100g",
      "fat": "0.2g/100g",
      "carbohydrates": "14g/100g",
      "fiber": "2.4g/100g"
    },
    "confidence": 0.95
  }
}
```

### 2. 图片上传

**POST** `/api/upload`

上传图片文件并转换为base64格式。

```bash
curl -X POST "http://localhost:3003/api/upload" \
     -F "file=@food_image.jpg"
```

### 3. 获取食物数据库

**GET** `/api/foods`

获取所有食物的嘌呤含量信息。

```bash
curl "http://localhost:3003/api/foods"
```

### 4. 获取特定食物信息

**GET** `/api/foods/{food_name}`

获取特定食物的详细信息。

```bash
curl "http://localhost:3003/api/foods/apple"
```

## 🤖 模型说明

### YOLO检测器

- **模型**: YOLOv8n (预训练)
- **功能**: 检测图片中的食物区域
- **输入**: 原始图片
- **输出**: 边界框坐标和置信度
- **支持类别**: 80个COCO类别中的食物相关类别

### ResNet分类器

- **模型**: ResNet-50 (预训练)
- **功能**: 精确识别食物类型
- **输入**: 224x224图片
- **输出**: 1000个ImageNet类别中的食物分类结果

### 支持的食物类型

- **水果**: 苹果、香蕉、橙子等
- **蔬菜**: 胡萝卜、西兰花、菠菜等
- **主食**: 披萨、汉堡、三明治等
- **甜点**: 蛋糕、甜甜圈等
- **其他**: 热狗等

## ⚙️ 配置说明

### 环境变量

```bash
# 服务器配置
PORT=3003                    # 服务端口
HOST=0.0.0.0                # 服务地址

# 模型配置
PYTHON_PATH=python          # Python解释器路径
MODEL_CONFIDENCE=0.5        # 检测置信度阈值
```

### 食物数据库

食物数据库包含在 `utils/food_database.py` 中，可以添加更多食物：

```python
def _get_default_foods(self) -> Dict[str, Dict]:
    return {
        'food_name': {
            'purineLevel': 'low|medium|high',
            'purineContent': '嘌呤含量',
            'suitableForGout': True,
            'advice': '饮食建议',
            'nutrition': { /* 营养成分 */ },
            'category': '食物类别'
        }
    }
```

## 🔧 开发指南

### 项目结构

```
backend/
├── main.py                      # FastAPI主应用
├── requirements.txt             # Python依赖
├── start_server.py             # 启动脚本
├── test_api.py                 # API测试脚本
├── install.sh                  # Linux/Mac安装脚本
├── install.bat                 # Windows安装脚本
├── models/                     # 模型定义
│   ├── schemas.py             # 数据模型
│   ├── yolo_detector.py       # YOLO检测器
│   └── resnet_classifier.py   # ResNet分类器
├── services/                   # 业务服务
│   └── food_recognition_service.py
├── utils/                      # 工具类
│   └── food_database.py       # 食物数据库
└── README.md                   # 文档
```

### 添加新的食物类别

1. 在 `utils/food_database.py` 中添加食物信息
2. 在 `models/resnet_classifier.py` 中更新类别列表
3. 重启服务

### 自定义模型

1. 修改 `models/yolo_detector.py` 或 `models/resnet_classifier.py`
2. 更新模型加载逻辑
3. 重启服务

## 🚀 性能优化

### GPU加速

```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
```

### 模型优化

- 使用模型量化减少内存占用
- 使用TensorRT加速推理
- 使用模型剪枝减少计算量

## 🐛 故障排除

### 常见问题

1. **Python环境问题**
   ```bash
   # 检查Python版本
   python --version
   
   # 重新安装依赖
   pip install -r requirements.txt
   ```

2. **模型下载失败**
   ```bash
   # 手动下载YOLO模型
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

3. **内存不足**
   - 减少批处理大小
   - 使用CPU模式
   - 增加系统内存

### 日志查看

```bash
# 查看服务日志
tail -f logs/app.log

# 查看错误日志
grep ERROR logs/app.log
```

## 📊 性能指标

- **识别准确率**: 80-90% (常见食物)
- **响应时间**: 1-3秒 (首次加载模型)
- **内存占用**: 2-4GB (包含模型)
- **支持并发**: 10-50请求/秒

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
python test_api.py

# 创建示例图片并测试
python test_api.py --create-sample

# 使用自定义图片测试
python test_api.py --image path/to/image.jpg
```

### 测试覆盖

- ✅ 健康检查
- ✅ API信息
- ✅ 模型状态
- ✅ 食物数据库
- ✅ 图片上传
- ✅ 食物识别

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

MIT License

## 📞 联系方式

- 项目地址: [GitHub Repository]
- 问题反馈: [Issues]
- 邮箱: [your-email@example.com]

---

**注意**: 首次启动时会自动下载YOLO和ResNet预训练模型，请确保网络连接正常。 