# 嘌呤识别助手 - AI服务配置指南

## 概述

本后端服务集成了阿里云通义千问VL模型，提供智能食物图像识别和嘌呤含量分析功能。

## 快速开始

### 1. 环境配置

#### 复制环境变量模板
```bash
cp env.example .env
```

#### 配置API Key
编辑 `.env` 文件，设置您的阿里云API Key：

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

### 2. 获取API Key

1. 访问 [阿里云DashScope控制台](https://dashscope.console.aliyun.com/)
2. 注册并登录阿里云账号
3. 在控制台中创建API Key
4. 复制API Key并配置到 `.env` 文件中

### 3. 安装依赖

```bash
npm install
```

### 4. 构建项目

```bash
npm run build
```

### 5. 测试AI服务

```bash
node test-ai.js
```

### 6. 启动服务

```bash
# 开发模式
npm run dev

# 生产模式
npm start
```

## 功能特性

### 智能食物识别
- 支持多种食物类型识别
- 准确识别食物名称和种类
- 高精度的图像分析能力

### 嘌呤含量分析
- 自动评估嘌呤含量等级（高/中/低）
- 提供具体的嘌呤含量数值
- 基于医学标准的分类

### 营养信息估算
- 热量、蛋白质、脂肪等营养成分
- 碳水化合物和膳食纤维含量
- 基于100g食物的标准化数据

### 健康建议
- 针对高尿酸患者的专业建议
- 详细的食用指导
- 个性化的饮食建议

## API接口

### 主要端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/info` | GET | 服务器信息 |
| `/api/ai-status` | GET | AI服务状态 |
| `/api/recognize` | POST | 食物识别 |
| `/api/upload` | POST | 图片上传 |
| `/api/foods` | GET | 食物列表 |

### 食物识别示例

```javascript
// 前端调用示例
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
```

## 返回数据格式

### 识别结果示例

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

## 嘌呤含量标准

### 分类标准
- **高嘌呤**: >150mg/100g - 痛风患者应避免食用
- **中嘌呤**: 50-150mg/100g - 痛风患者应适量食用  
- **低嘌呤**: <50mg/100g - 痛风患者可以放心食用

### 常见食物分类

#### 高嘌呤食物
- 动物肝脏、肾脏
- 海鲜类（虾、蟹、贝类）
- 浓肉汤、肉汁
- 啤酒、白酒

#### 中嘌呤食物
- 肉类（牛肉、猪肉、羊肉）
- 豆类、豆腐
- 蘑菇、香菇
- 菠菜、芦笋

#### 低嘌呤食物
- 大部分蔬菜
- 水果类
- 牛奶、鸡蛋
- 谷物类

## 故障排除

### 常见问题

#### 1. AI服务初始化失败
**错误信息**: `DASHSCOPE_API_KEY 环境变量未设置`

**解决方案**:
- 检查 `.env` 文件是否存在
- 确认 `DASHSCOPE_API_KEY` 已正确设置
- 重启服务

#### 2. API调用失败
**错误信息**: `401 Unauthorized`

**解决方案**:
- 检查API Key是否正确
- 确认API Key有足够的配额
- 验证API Key是否已激活

#### 3. 网络连接问题
**错误信息**: `Network Error`

**解决方案**:
- 检查网络连接
- 确认防火墙设置
- 验证代理配置

### 调试步骤

1. **检查环境变量**
```bash
node -e "console.log(require('dotenv').config())"
```

2. **测试AI服务**
```bash
node test-ai.js
```

3. **检查服务状态**
```bash
curl http://localhost:3001/api/ai-status
```

4. **查看服务日志**
```bash
npm run dev
```

## 性能优化

### 建议配置

1. **图片大小**: 建议不超过8MB
2. **图片格式**: 推荐JPEG或PNG
3. **并发请求**: 根据API配额合理控制
4. **缓存策略**: 可考虑对常见食物进行缓存

### 监控指标

- API响应时间
- 识别准确率
- 错误率统计
- 资源使用情况

## 安全注意事项

1. **API Key保护**: 不要将API Key提交到版本控制系统
2. **环境隔离**: 开发和生产环境使用不同的API Key
3. **访问控制**: 在生产环境中实施适当的访问控制
4. **日志安全**: 避免在日志中记录敏感信息

## 更新日志

### v1.0.0 (2024-01-15)
- 初始版本发布
- 集成通义千问VL模型
- 支持食物图像识别
- 提供完整的营养分析功能

## 技术支持

如遇到问题，请检查：
1. 环境变量配置
2. API Key有效性
3. 网络连接状态
4. 服务日志信息

更多详细信息请参考 `api.md` 文档。 