
# 嘟呤识别助手

本项目是一个基于 YOLO + ResNet50 融合模型的食物识别与尿酸分析系统，后端采用 FastAPI 框架，前端基于 Next.js。支持拍照或上传图片自动识别食物类别，分析嘌呤含量并给出痛风饮食建议。

## 主要功能

- 🍽️ **食物识别**：拍照或上传图片，自动检测食物区域并分类
- 🧬 **尿酸分析**：结合食物数据库，输出嘌呤含量及痛风建议
- 📊 **营养估算**：展示热量、蛋白质、脂肪等营养信息
- 🎯 **饮食建议**：针对痛风患者的个性化推荐
- 🚀 **高性能推理**：融合 YOLOv8 检测与 ResNet50 分类，支持 GPU 加速

## 工作原理

- 采用 **前后端分离架构设计开发**，前端通过 **Next.js** 构建用户界面，后端基于 **FastAPI** 提供 RESTful API 接口；
- 系统核心功能拆分为：图像处理服务、食物识别服务、数据分析服务、用户界面服务、模型推理服务；
- 基于 **YOLOv8** + **ResNet50** 融合模型实现食物识别，YOLOv8 负责目标检测，ResNet50 负责图像分类，通过 **OpenCV** 进行图像预处理；
- 设计并实现 **食物数据库查询机制**，结合识别结果自动匹配食物嘌呤含量及营养信息；
- 基于 **React Zustand** 实现状态管理，通过 **Tailwind CSS** 构建响应式用户界面，提升用户体验；
- 通过 **Docker** 容器化部署前后端服务，解决环境配置复杂和部署困难的问题。

## 技术架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 前端应用     │◄──►│ FastAPI后端 │◄──►│  AI模型层   │
│ Next.js/TS  │    │ API服务     │    │ YOLO+ResNet │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 数据集与模型训练

- 使用 Food20_new 数据集（20类食物，约400张/类）
- 自动转换为 YOLO 格式，支持目标检测与分类
- 训练脚本：`backend/scripts/train_food20_models.py`
- 支持自定义训练参数与数据增强
- 输出模型：
  - `models/resnet_food20_classifier.pth`（ResNet分类器）
  - `models/yolo_food20_detector.pt`（YOLO检测器）

### 数据集结构
```
datasets/Food20_new/
├── train/
│   ├── images/
│   └── train.json
├── val/
│   ├── images/
│   └── val.json
└── test/
    ├── images/
    └── test.json
```

### 训练命令示例
```bash
# 安装依赖
pip install -r backend/requirements.txt

# 训练ResNet分类器
python backend/scripts/train_food20_models.py --model resnet

# 训练YOLO检测器
python backend/scripts/train_food20_models.py --model yolo

# 评估模型
python backend/scripts/train_food20_models.py --evaluate
```

## 快速开始

### 环境要求
- Node.js 16+
- Python 3.8+
- 推荐GPU

### 安装依赖
```bash
# 前端依赖
npm install
# 后端依赖
cd backend
pip install -r requirements.txt
```

### 环境变量配置
前端：复制 `env.example` 为 `.env.local`，配置后端API地址。
后端：复制 `env.example` 为 `.env`，配置端口等参数。

### 启动服务
```bash
# 启动后端
cd backend
python main.py
# 启动前端
npm run dev
```
前端：http://localhost:3000
后端：http://localhost:3003


## 项目结构

```
project/
├── app/                    # Next.js前端应用
│   ├── page.tsx           # 主页面
│   ├── layout.tsx         # 布局组件
│   └── test-camera/       # 相机测试页面
├── components/            # React组件
│   ├── ui/               # 基础UI组件
│   ├── CameraCapture.tsx # 相机拍照组件
│   ├── RecognitionPage.tsx # 识别页面
│   └── ...
├── lib/                  # 工具库
│   └── recognition-api.ts # 识别API服务
├── backend/              # 后端服务
│   ├── main.py           # FastAPI主应用
│   ├── models/           # AI模型
│   ├── services/         # 业务服务
│   ├── utils/            # 工具类
|   ├── datasets/         # 训练模型数据集
|   ├── scripts/          # 训练YOLO+ResNet50 
│   └── requirements.txt  # Python依赖
├── package.json          # 前端依赖
└── README.md            # 项目说明
```

## API端点

### 后端API (http://localhost:3003)

- `GET /health` - 健康检查
- `GET /api/info` - 服务器信息
- `GET /api/model-status` - 模型状态
- `POST /api/recognize` - 食物识别
- `POST /api/upload` - 图片上传
- `GET /api/foods` - 食物数据库
- `GET /api/foods/{food_name}` - 特定食物信息

### 前端页面 (http://localhost:3000)

- `/` - 主应用页面
- `/test-camera` - 相机功能测试页面


## 使用说明

### 拍照识别
1. 点击“拍照识别”按钮
2. 允许浏览器访问相机
3. 将食物放在取景框内
4. 拍照并确认
5. 自动识别并显示结果

### 图片上传
1. 点击“从相册选择”按钮
2. 选择食物图片
3. 点击“开始识别”
4. 显示识别结果与分析


## 技术栈

### 前端
- Next.js 13 + React 18
- Radix UI + Tailwind CSS
- Zustand 状态管理
- TypeScript 类型安全
- react-webcam 相机功能

### 后端
- FastAPI 框架
- Python 3.8+
- YOLOv8 检测器 + ResNet50 分类器
- OpenCV, Pillow 图像处理
- PyTorch, Ultralytics 机器学习

### 训练与推理
- 支持 GPU 加速
- 自动数据增强与模型保存
- API接口统一，易于扩展


## 开发指南

### 前端开发
```bash
npm run dev          # 开发模式
npm run build        # 构建生产版本
npm run start        # 启动生产服务器
```

### 后端开发
```bash
cd backend
python start_server.py    # 开发模式（热重载）
uvicorn main:app --reload # 直接使用uvicorn
```

### 测试
```bash
# 集成测试
node test-integration.js
# 后端API测试
cd backend
python test_api.py
```


## 部署

### 前端部署
```bash
npm run build
npm run start
```

### 后端部署
```bash
cd backend
python start_server.py
```

### Docker 部署
```bash
# 前端
docker build -t purine-frontend .
docker run -p 3000:3000 purine-frontend
# 后端
cd backend
docker build -t purine-backend .
docker run -p 3003:3003 purine-backend
```


## 环境变量

### 前端 (.env.local)
```env
# 后端API配置
NEXT_PUBLIC_BACKEND_URL=http://localhost:3003
```

### 后端 (.env)
```env
# 服务器配置
PORT=3003
HOST=0.0.0.0
# 环境配置
NODE_ENV=development
```


## 故障排除

### 相机无法使用
- 检查设备摄像头和浏览器权限
- 建议生产环境使用 HTTPS

### 后端服务无法连接
- 检查后端服务是否启动
- 确认端口未被占用
- 检查防火墙设置

### 识别失败
- 检查网络连接
- 检查API密钥和模型文件
- 查看后端/前端日志




