# 嘌呤识别助手

基于AI的食物识别和尿酸含量分析应用，帮助痛风患者科学饮食。

## 功能特性

- 🍽️ **智能识别**: 拍照或上传图片，AI自动识别食物
- 🧬 **尿酸分析**: 提供详细的嘌呤含量信息
- 📊 **营养估算**: 计算食物的营养成分
- 🎯 **饮食建议**: 针对痛风患者的专业建议
- 📱 **移动友好**: 响应式设计，支持手机端使用
- 🚀 **高性能**: 基于YOLO+ResNet融合模型

## 快速开始

### 1. 环境要求

- Node.js 16+
- Python 3.8+
- 至少4GB内存

### 2. 安装依赖

```bash
# 安装前端依赖
npm install

# 安装后端依赖
cd backend
pip install -r requirements.txt
```

### 3. 配置环境变量

#### 前端配置
复制 `env.example` 为 `.env.local`：
```bash
cp env.example .env.local
```

编辑 `.env.local` 文件：
```env
# 后端API配置
NEXT_PUBLIC_BACKEND_URL=http://localhost:3003


```

#### 后端配置
复制 `backend/env.example` 为 `backend/.env`：
```bash
cd backend
cp env.example .env
```

编辑 `backend/.env` 文件：
```env
# 服务器配置
PORT=3003
HOST=0.0.0.0

# 环境配置
NODE_ENV=development


```

### 4. 启动服务


```bash
# 终端1：启动后端
cd backend
python main.py

# 终端2：启动前端
npm run dev
```


前端服务运行在 `http://localhost:3000`
后端服务运行在 `http://localhost:3003`

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
1. 点击"拍照识别"按钮
2. 允许浏览器访问相机
3. 将食物放在取景框内
4. 点击拍照按钮
5. 确认照片后开始识别

### 图片上传
1. 点击"从相册选择"按钮
2. 选择要识别的食物图片
3. 点击"开始识别"按钮

## 技术栈

### 前端
- **框架**: Next.js 13 + React 18
- **UI组件**: Radix UI + Tailwind CSS
- **相机功能**: react-webcam
- **状态管理**: Zustand
- **类型安全**: TypeScript

### 后端
- **框架**: FastAPI
- **语言**: Python
- **AI模型**: YOLO + ResNet
- **图像处理**: OpenCV, Pillow
- **机器学习**: PyTorch, Ultralytics

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
# 测试前后端集成
node test-integration.js

# 测试后端API
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

### Docker部署
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
- 确保设备有摄像头
- 检查浏览器权限设置
- 确保使用HTTPS环境（生产环境）

### 后端服务无法连接
- 检查后端服务是否启动
- 确认端口3003未被占用
- 检查防火墙设置

### 识别失败
- 检查网络连接
- 确认API密钥配置正确
- 查看浏览器控制台错误信息


