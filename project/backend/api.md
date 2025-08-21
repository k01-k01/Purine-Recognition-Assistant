# 嘌呤识别助手后端API文档

## 概述

本API服务基于Fastify框架构建，提供食物图像识别和嘌呤含量分析功能。服务集成了阿里云通义千问VL模型，能够准确识别食物并提供详细的营养信息。

## 基础信息

- **服务地址**: `http://localhost:3001`
- **API版本**: v1.0.0
- **数据格式**: JSON
- **字符编码**: UTF-8

## 环境配置

### 必需的环境变量

在 `backend/.env` 文件中配置以下环境变量：

```env
# 阿里云通义千问API配置
DASHSCOPE_API_KEY=your_api_key_here

# 服务器配置
PORT=3001
HOST=0.0.0.0

# 可选：模型配置
AI_MODEL=qwen-vl-plus
AI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 获取API Key

1. 访问 [阿里云DashScope控制台](https://dashscope.console.aliyun.com/)
2. 注册并登录账号
3. 创建API Key
4. 将API Key配置到环境变量中

## API端点

### 1. 健康检查

检查服务器运行状态。

**请求**
```http
GET /health
```

**响应**
```json
{
  "status": "ok",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### 2. 服务器信息

获取服务器详细信息和可用端点。

**请求**
```http
GET /api/info
```

**响应**
```json
{
  "name": "嘌呤识别助手后端服务",
  "version": "1.0.0",
  "description": "基于Fastify的食物嘌呤识别API服务",
  "aiService": {
    "status": "healthy",
    "model": "qwen-vl-plus",
    "available": true
  },
  "endpoints": [
    "GET /health - 健康检查",
    "GET /api/info - 服务器信息",
    "GET /api/ai-status - AI服务状态",
    "POST /api/recognize - 食物识别",
    "POST /api/upload - 图片上传",
    "GET /api/foods - 食物列表"
  ]
}
```

### 3. AI服务状态

检查AI服务的可用性和状态。

**请求**
```http
GET /api/ai-status
```

**响应**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "model": "qwen-vl-plus",
    "available": true
  }
}
```

### 4. 食物识别

识别上传的食物图像并返回详细的营养信息。

**请求**
```http
POST /api/recognize
Content-Type: application/json
```

**请求体**
```json
{
  "image": "base64编码的图片数据"
}
```

**响应**
```json
{
  "success": true,
  "data": {
    "foodName": "三文鱼",
    "purineLevel": "high",
    "purineContent": "150-200mg/100g",
    "suitableForGout": false,
    "advice": "三文鱼属于高嘌呤食物，每100g含有150-200mg嘌呤。高尿酸患者应避免食用，建议选择低嘌呤的鱼类如鲈鱼或鳕鱼。",
    "nutritionEstimate": {
      "calories": "208kcal/100g",
      "protein": "22g/100g",
      "fat": "13g/100g",
      "carbohydrates": "0g/100g",
      "fiber": "0g/100g"
    },
    "confidence": 0.95
  }
}
```

**错误响应**
```json
{
  "success": false,
  "error": "错误信息"
}
```

### 5. 图片上传

上传图片文件并返回base64编码。

**请求**
```http
POST /api/upload
Content-Type: multipart/form-data
```

**请求参数**
- `file`: 图片文件（支持jpg, png, gif等格式）

**响应**
```json
{
  "success": true,
  "data": {
    "filename": "food.jpg",
    "mimetype": "image/jpeg",
    "size": 1024000,
    "dataUrl": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
  }
}
```

### 6. 食物列表

获取预定义的食物嘌呤信息列表。

**请求**
```http
GET /api/foods
```

**响应**
```json
{
  "success": true,
  "data": [
    {
      "name": "动物肝脏",
      "level": "high",
      "content": "150-1000mg/100g",
      "advice": "高嘌呤食物，痛风患者应避免食用"
    },
    {
      "name": "海鲜",
      "level": "high",
      "content": "100-300mg/100g",
      "advice": "高嘌呤食物，需要限制摄入"
    }
  ]
}
```

## 数据模型

### FoodRecognitionResult

食物识别结果的数据结构：

```typescript
interface FoodRecognitionResult {
  foodName: string;                    // 食物名称
  purineLevel: 'high' | 'medium' | 'low'; // 嘌呤含量等级
  purineContent: string;               // 具体嘌呤含量
  suitableForGout: boolean;            // 是否适合高尿酸患者
  advice: string;                      // 食用建议
  nutritionEstimate: {                 // 营养成分估算
    calories: string;                  // 热量
    protein: string;                   // 蛋白质
    fat: string;                       // 脂肪
    carbohydrates: string;             // 碳水化合物
    fiber: string;                     // 膳食纤维
  };
  confidence: number;                  // 识别置信度 (0-1)
}
```

### 嘌呤含量等级标准

- **高嘌呤**: >150mg/100g - 痛风患者应避免食用
- **中嘌呤**: 50-150mg/100g - 痛风患者应适量食用
- **低嘌呤**: <50mg/100g - 痛风患者可以放心食用

## 错误处理

### 常见错误码

| 错误类型 | 描述 | 解决方案 |
|---------|------|----------|
| 400 | 请求参数错误 | 检查请求体格式和必需字段 |
| 401 | 未授权 | 检查API Key配置 |
| 500 | 服务器内部错误 | 检查服务器日志 |
| AI服务不可用 | AI服务初始化失败 | 检查环境变量配置 |

### 错误响应格式

```json
{
  "success": false,
  "error": "详细的错误信息"
}
```

## 使用示例

### JavaScript/TypeScript

```javascript
// 食物识别示例
async function recognizeFood(imageBase64) {
  const response = await fetch('http://localhost:3001/api/recognize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: imageBase64
    })
  });
  
  const result = await response.json();
  return result;
}

// 图片上传示例
async function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('http://localhost:3001/api/upload', {
    method: 'POST',
    body: formData
  });
  
  const result = await response.json();
  return result;
}
```

### cURL

```bash
# 健康检查
curl http://localhost:3001/health

# 服务器信息
curl http://localhost:3001/api/info

# AI服务状态
curl http://localhost:3001/api/ai-status

# 食物识别
curl -X POST http://localhost:3001/api/recognize \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_data"}'

# 图片上传
curl -X POST http://localhost:3001/api/upload \
  -F "file=@food.jpg"
```

## 部署说明

### 开发环境

```bash
cd backend
npm install
npm run dev
```

### 生产环境

```bash
cd backend
npm install
npm run build
npm start
```

### Docker部署

```bash
# 构建镜像
docker build -t purine-recognition-backend .

# 运行容器
docker run -p 3001:3001 \
  -e DASHSCOPE_API_KEY=your_api_key \
  purine-recognition-backend
```

## 注意事项

1. **API Key安全**: 请妥善保管API Key，不要提交到版本控制系统
2. **图片格式**: 支持常见的图片格式，建议使用JPEG或PNG
3. **图片大小**: 建议图片大小不超过8MB
4. **请求频率**: 请合理控制请求频率，避免超出API配额
5. **错误处理**: 建议在客户端实现适当的错误处理和重试机制

## 更新日志

### v1.0.0 (2024-01-15)
- 初始版本发布
- 集成通义千问VL模型
- 支持食物图像识别和营养分析
- 提供完整的RESTful API接口 