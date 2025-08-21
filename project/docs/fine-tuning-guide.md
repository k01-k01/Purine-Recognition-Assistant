# 通义千问VL模型微调指南

## 概述

本指南将帮助您对通义千问VL模型进行微调，以提高食物识别的准确性。微调可以让模型更好地理解特定领域的食物特征和分类。

## 微调方案选择

### 1. 阿里云官方微调服务（推荐）

阿里云提供了专门的模型微调服务，支持通义千问VL模型：

#### 优势：
- 官方支持，稳定性高
- 自动处理训练环境
- 提供完整的训练流程
- 支持增量训练

#### 步骤：

1. **准备训练数据**
   ```bash
   # 创建训练数据目录
   mkdir -p training_data
   cd training_data
   ```

2. **数据格式要求**
   ```json
   {
     "conversations": [
       {
         "role": "user",
         "content": [
           {
             "type": "image",
             "image": "base64编码的图片"
           },
           {
             "type": "text", 
             "text": "请识别这张图片中的食物"
           }
         ]
       },
       {
         "role": "assistant",
         "content": "这是三文鱼，属于高嘌呤食物，每100g含有150-200mg嘌呤。高尿酸患者应避免食用。"
       }
     ]
   }
   ```

3. **数据收集策略**
   - 使用项目中的食物图片库
   - 收集不同角度、光线条件下的图片
   - 包含相似食物的对比图片
   - 添加边界情况的图片

### 2. 自建微调环境

如果您有足够的计算资源，可以自建微调环境：

#### 硬件要求：
- GPU: 至少16GB显存（推荐A100或V100）
- 内存: 64GB以上
- 存储: 1TB SSD

#### 软件环境：
```bash
# 安装依赖
pip install torch torchvision
pip install transformers datasets
pip install accelerate peft
```

## 数据准备

### 1. 数据收集

基于您现有的食物图片库，创建训练数据集：

```python
import os
import json
import base64
from PIL import Image
import io

def create_training_data():
    """创建训练数据"""
    training_data = []
    
    # 食物图片目录
    img_dir = "project/public/imgs"
    
    # 食物数据库
    foods_data = load_foods_data()
    
    for filename in os.listdir(img_dir):
        if filename.endswith('.jpg'):
            food_name = filename.replace('.jpg', '')
            
            # 查找对应的食物信息
            food_info = find_food_info(food_name, foods_data)
            
            if food_info:
                # 读取图片并转换为base64
                img_path = os.path.join(img_dir, filename)
                with open(img_path, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode()
                
                # 创建训练样本
                sample = {
                    "conversations": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": img_base64
                                },
                                {
                                    "type": "text",
                                    "text": "请识别这张图片中的食物并提供嘌呤含量信息"
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": create_assistant_response(food_info)
                        }
                    ]
                }
                
                training_data.append(sample)
    
    return training_data

def create_assistant_response(food_info):
    """创建助手回复"""
    return f"""这是{food_info['name']}，属于{food_info['purineLevel']}嘌呤食物，每100g含有{food_info['purineContent']}嘌呤。

{food_info['description']}

营养信息：
- 热量：约{food_info.get('calories', '100-150')}kcal/100g
- 蛋白质：约{food_info.get('protein', '10-20')}g/100g
- 脂肪：约{food_info.get('fat', '1-5')}g/100g

建议：{get_advice(food_info['purineLevel'])}"""

def get_advice(purine_level):
    """根据嘌呤等级给出建议"""
    if purine_level == 'high':
        return "高尿酸患者应避免食用，建议选择低嘌呤的替代食物。"
    elif purine_level == 'medium':
        return "高尿酸患者应限制食用，每周不超过2-3次。"
    else:
        return "高尿酸患者可以适量食用，但仍需注意控制总量。"
```

### 2. 数据增强

为了提高模型的泛化能力，建议进行数据增强：

```python
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def augment_image(image_path, output_dir):
    """图像数据增强"""
    img = Image.open(image_path)
    
    # 1. 亮度调整
    enhancer = ImageEnhance.Brightness(img)
    bright_img = enhancer.enhance(1.3)
    bright_img.save(f"{output_dir}/bright_{os.path.basename(image_path)}")
    
    # 2. 对比度调整
    enhancer = ImageEnhance.Contrast(img)
    contrast_img = enhancer.enhance(1.2)
    contrast_img.save(f"{output_dir}/contrast_{os.path.basename(image_path)}")
    
    # 3. 旋转
    rotated_img = img.rotate(15)
    rotated_img.save(f"{output_dir}/rotated_{os.path.basename(image_path)}")
    
    # 4. 翻转
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_img.save(f"{output_dir}/flipped_{os.path.basename(image_path)}")
```

### 3. 数据验证

确保训练数据的质量：

```python
def validate_training_data(training_data):
    """验证训练数据质量"""
    issues = []
    
    for i, sample in enumerate(training_data):
        # 检查图片格式
        try:
            img_data = sample['conversations'][0]['content'][0]['image']
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
            
            # 检查图片尺寸
            if img.size[0] < 100 or img.size[1] < 100:
                issues.append(f"样本{i}: 图片尺寸过小 {img.size}")
                
        except Exception as e:
            issues.append(f"样本{i}: 图片格式错误 {e}")
        
        # 检查回复内容
        assistant_content = sample['conversations'][1]['content']
        if len(assistant_content) < 50:
            issues.append(f"样本{i}: 回复内容过短")
    
    return issues
```

## 微调流程

### 1. 使用阿里云微调服务

```python
import dashscope
from dashscope import FineTune

def start_fine_tuning():
    """启动微调任务"""
    
    # 上传训练数据
    response = FineTune.upload_file(
        file_path='training_data.json',
        file_type='json'
    )
    
    if response.status_code == 200:
        file_id = response.output.file_id
        
        # 创建微调任务
        response = FineTune.create(
            model='qwen-vl-plus',
            file_id=file_id,
            hyper_parameters={
                'learning_rate': 1e-5,
                'epochs': 3,
                'batch_size': 4
            }
        )
        
        if response.status_code == 200:
            job_id = response.output.job_id
            print(f"微调任务已创建，任务ID: {job_id}")
            return job_id
    
    return None

def check_fine_tuning_status(job_id):
    """检查微调状态"""
    response = FineTune.get(job_id=job_id)
    
    if response.status_code == 200:
        status = response.output.status
        print(f"微调状态: {status}")
        return status
    
    return None
```

### 2. 自建微调环境

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

def setup_fine_tuning():
    """设置微调环境"""
    
    # 加载模型和分词器
    model_name = "Qwen/Qwen-VL-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def train_model(model, tokenizer, training_data):
    """训练模型"""
    from transformers import TrainingArguments, Trainer
    
    # 准备训练数据
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512
        )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
```

## 模型评估

### 1. 评估指标

```python
def evaluate_model(model, test_data):
    """评估模型性能"""
    
    metrics = {
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0
    }
    
    correct_predictions = 0
    total_predictions = len(test_data)
    
    for sample in test_data:
        # 获取预测结果
        prediction = model.predict(sample['input'])
        
        # 与真实标签比较
        if prediction['foodName'] == sample['expected']['foodName']:
            correct_predictions += 1
    
    metrics['accuracy'] = correct_predictions / total_predictions
    
    return metrics
```

### 2. 对比测试

```python
def compare_models(original_model, fine_tuned_model, test_cases):
    """对比原始模型和微调后的模型"""
    
    results = {
        'original': [],
        'fine_tuned': []
    }
    
    for test_case in test_cases:
        # 测试原始模型
        original_result = original_model.recognize(test_case['image'])
        results['original'].append({
            'expected': test_case['expected'],
            'predicted': original_result,
            'correct': original_result['foodName'] == test_case['expected']['foodName']
        })
        
        # 测试微调模型
        fine_tuned_result = fine_tuned_model.recognize(test_case['image'])
        results['fine_tuned'].append({
            'expected': test_case['expected'],
            'predicted': fine_tuned_result,
            'correct': fine_tuned_result['foodName'] == test_case['expected']['foodName']
        })
    
    return results
```

## 部署微调后的模型

### 1. 更新AI服务配置

```typescript
// 在 ai-service.ts 中添加微调模型支持
export class AIService {
  private client: OpenAI;
  private model: string;
  private fineTunedModel?: string;

  constructor() {
    const apiKey = process.env.DASHSCOPE_API_KEY;
    const baseURL = process.env.AI_BASE_URL || 'https://dashscope.aliyuncs.com/compatible-mode/v1';
    this.model = process.env.AI_MODEL || 'qwen-vl-plus';
    this.fineTunedModel = process.env.FINE_TUNED_MODEL; // 微调后的模型ID

    if (!apiKey) {
      throw new Error('DASHSCOPE_API_KEY 环境变量未设置');
    }

    this.client = new OpenAI({
      apiKey,
      baseURL,
    });
  }

  async recognizeFood(base64Image: string): Promise<FoodRecognitionResult> {
    try {
      // 优先使用微调后的模型
      const modelToUse = this.fineTunedModel || this.model;
      
      const response = await this.client.chat.completions.create({
        model: modelToUse,
        messages: [
          {
            role: 'system',
            content: systemPrompt,
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: '请分析这张食物图片，识别食物并提供详细的营养信息和建议。',
              },
              {
                type: 'image_url',
                image_url: {
                  url: `data:image/jpeg;base64,${base64Image}`,
                },
              },
            ],
          },
        ],
        response_format: { type: 'json_object' },
        temperature: 0.1,
        max_tokens: 1000,
      });

      // ... 其余代码保持不变
    } catch (error) {
      console.error('AI识别失败:', error);
      throw new Error(`食物识别失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  }
}
```

### 2. 环境变量配置

```env
# 微调模型配置
FINE_TUNED_MODEL=your_fine_tuned_model_id
FINE_TUNED_ENABLED=true
```

## 成本估算

### 阿里云微调服务成本

1. **数据准备**: 免费
2. **微调训练**: 
   - 基础版: 约500-1000元/次
   - 高级版: 约1000-2000元/次
3. **模型推理**: 按使用量计费，约0.1-0.2元/次调用

### 自建环境成本

1. **硬件成本**: 
   - GPU服务器: 约5000-10000元/月
   - 存储: 约500元/月
2. **时间成本**: 约1-2周开发和调试

## 建议的实施步骤

### 第一阶段：优化提示词（立即实施）
1. 使用优化后的系统提示词
2. 测试识别效果
3. 收集用户反馈

### 第二阶段：数据准备（1-2周）
1. 整理现有食物图片库
2. 创建训练数据集
3. 进行数据增强
4. 验证数据质量

### 第三阶段：模型微调（2-4周）
1. 选择微调方案
2. 执行微调训练
3. 评估模型性能
4. 部署微调后的模型

### 第四阶段：持续优化（持续）
1. 收集用户反馈
2. 扩充训练数据
3. 定期重新微调
4. 监控模型性能

## 注意事项

1. **数据隐私**: 确保训练数据不包含敏感信息
2. **模型版本管理**: 保留原始模型作为备份
3. **性能监控**: 持续监控微调后模型的性能
4. **成本控制**: 根据实际需求选择合适的微调方案
5. **合规性**: 确保微调过程符合相关法规要求

## 技术支持

如果在微调过程中遇到问题，可以：

1. 查看阿里云官方文档
2. 联系阿里云技术支持
3. 参考开源社区资源
4. 寻求专业AI服务商帮助 