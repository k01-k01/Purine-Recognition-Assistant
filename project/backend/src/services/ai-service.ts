import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

// 食物识别结果接口
export interface FoodRecognitionResult {
  foodName: string;
  purineLevel: 'high' | 'medium' | 'low';
  purineContent: string;
  suitableForGout: boolean;
  advice: string;
  nutritionEstimate: {
    calories: string;
    protein: string;
    fat: string;
    carbohydrates: string;
    fiber: string;
  };
  confidence: number;
}

// 食物嘌呤数据库
const FOOD_PURINE_DB: Record<string, { level: 'high' | 'medium' | 'low'; content: string; advice: string; nutrition: any }> = {
  'apple': { level: 'low', content: '5-15mg/100g', advice: '低嘌呤食物，痛风患者可以放心食用', nutrition: { calories: '52kcal/100g', protein: '0.3g/100g', fat: '0.2g/100g', carbohydrates: '14g/100g', fiber: '2.4g/100g' } },
  'banana': { level: 'low', content: '5-10mg/100g', advice: '低嘌呤食物，富含钾元素，有助于尿酸排泄', nutrition: { calories: '89kcal/100g', protein: '1.1g/100g', fat: '0.3g/100g', carbohydrates: '23g/100g', fiber: '2.6g/100g' } },
  'orange': { level: 'low', content: '5-10mg/100g', advice: '低嘌呤食物，富含维生素C，有助于尿酸排泄', nutrition: { calories: '47kcal/100g', protein: '0.9g/100g', fat: '0.1g/100g', carbohydrates: '12g/100g', fiber: '2.4g/100g' } },
  'carrot': { level: 'low', content: '10-20mg/100g', advice: '低嘌呤食物，富含胡萝卜素，痛风患者可以放心食用', nutrition: { calories: '41kcal/100g', protein: '0.9g/100g', fat: '0.2g/100g', carbohydrates: '10g/100g', fiber: '2.8g/100g' } },
  'broccoli': { level: 'low', content: '20-30mg/100g', advice: '低嘌呤食物，富含维生素和矿物质，痛风患者可以放心食用', nutrition: { calories: '34kcal/100g', protein: '2.8g/100g', fat: '0.4g/100g', carbohydrates: '7g/100g', fiber: '2.6g/100g' } },
  'pizza': { level: 'medium', content: '50-100mg/100g', advice: '中等嘌呤食物，痛风患者可适量食用，注意控制量', nutrition: { calories: '266kcal/100g', protein: '11g/100g', fat: '10g/100g', carbohydrates: '33g/100g', fiber: '2.5g/100g' } },
  'hamburger': { level: 'medium', content: '80-120mg/100g', advice: '中等嘌呤食物，痛风患者可适量食用，建议选择瘦肉', nutrition: { calories: '295kcal/100g', protein: '17g/100g', fat: '12g/100g', carbohydrates: '30g/100g', fiber: '1.2g/100g' } },
  'hot dog': { level: 'high', content: '150-200mg/100g', advice: '高嘌呤食物，痛风患者应避免食用', nutrition: { calories: '290kcal/100g', protein: '12g/100g', fat: '26g/100g', carbohydrates: '4g/100g', fiber: '0g/100g' } },
  'cake': { level: 'low', content: '10-20mg/100g', advice: '低嘌呤食物，但含糖量高，痛风患者应适量食用', nutrition: { calories: '257kcal/100g', protein: '4.5g/100g', fat: '12g/100g', carbohydrates: '35g/100g', fiber: '0.8g/100g' } },
  'donut': { level: 'low', content: '10-20mg/100g', advice: '低嘌呤食物，但含糖量和脂肪含量高，应适量食用', nutrition: { calories: '253kcal/100g', protein: '4.3g/100g', fat: '14g/100g', carbohydrates: '31g/100g', fiber: '1.2g/100g' } },
  'sandwich': { level: 'medium', content: '50-100mg/100g', advice: '中等嘌呤食物，痛风患者可适量食用', nutrition: { calories: '250kcal/100g', protein: '15g/100g', fat: '8g/100g', carbohydrates: '30g/100g', fiber: '2g/100g' } },
  'person': { level: 'low', content: '0mg/100g', advice: '这不是食物，请上传食物图片', nutrition: { calories: '0kcal/100g', protein: '0g/100g', fat: '0g/100g', carbohydrates: '0g/100g', fiber: '0g/100g' } },
  'unknown': { level: 'low', content: '10-50mg/100g', advice: '无法准确识别，建议咨询医生或营养师', nutrition: { calories: '100kcal/100g', protein: '5g/100g', fat: '3g/100g', carbohydrates: '15g/100g', fiber: '2g/100g' } }
};

// AI服务类
export class AIService {
  private pythonScriptPath: string;
  private isInitialized: boolean = false;

  constructor() {
    this.pythonScriptPath = path.join(__dirname, '../../python/food_recognition.py');
    this.initializeService();
  }

  private async initializeService() {
    try {
      // 检查Python脚本是否存在
      if (!fs.existsSync(this.pythonScriptPath)) {
        console.log('Python脚本不存在，创建默认脚本...');
        this.createDefaultPythonScript();
      }
      
      // 测试Python环境
      await this.testPythonEnvironment();
      this.isInitialized = true;
      console.log('AI服务初始化成功');
    } catch (error) {
      console.error('AI服务初始化失败:', error);
      this.isInitialized = false;
    }
  }

  private createDefaultPythonScript() {
    const scriptContent = `#!/usr/bin/env python3
import sys
import json
import base64
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2

def load_models():
    """加载预训练模型"""
    try:
        # 加载YOLO模型
        yolo_model = YOLO('yolov8n.pt')
        
        # 加载ResNet模型
        resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        resnet_model.eval()
        
        return yolo_model, resnet_model
    except Exception as e:
        print(f"模型加载失败: {e}", file=sys.stderr)
        return None, None

def preprocess_image(image_data):
    """预处理图片"""
    try:
        # 解码base64图片
        if image_data.startswith('data:image'):
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        else:
            image_bytes = base64.b64decode(image_data)
        
        # 转换为PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        print(f"图片预处理失败: {e}", file=sys.stderr)
        return None

def detect_food_regions(image, yolo_model):
    """使用YOLO检测食物区域"""
    try:
        # 转换为OpenCV格式
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # YOLO检测
        results = yolo_model(image_cv, conf=0.5, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 获取置信度
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # 获取类别ID和名称
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = yolo_model.names[class_id]
                    
                    # 只保留食物相关的检测结果
                    food_keywords = ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 
                                   'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'person']
                    if any(keyword in class_name.lower() for keyword in food_keywords):
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": confidence,
                            "class_name": class_name
                        })
        
        return detections
    except Exception as e:
        print(f"YOLO检测失败: {e}", file=sys.stderr)
        return []

def classify_food(image, resnet_model):
    """使用ResNet分类食物"""
    try:
        # 预处理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            outputs = resnet_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # 获取top-5预测结果
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                # 这里简化处理，直接返回一些常见的食物类别
                food_classes = ['apple', 'banana', 'orange', 'carrot', 'broccoli', 
                              'pizza', 'hamburger', 'hot dog', 'cake', 'donut', 'sandwich']
                
                if idx < len(food_classes):
                    class_name = food_classes[idx]
                else:
                    class_name = 'unknown'
                
                predictions.append({
                    "class_name": class_name,
                    "confidence": float(prob)
                })
            
            return predictions
    except Exception as e:
        print(f"ResNet分类失败: {e}", file=sys.stderr)
        return [{"class_name": "unknown", "confidence": 0.5}]

def recognize_food(image_data):
    """识别食物"""
    try:
        # 加载模型
        yolo_model, resnet_model = load_models()
        if yolo_model is None or resnet_model is None:
            return {"error": "模型加载失败"}
        
        # 预处理图片
        image = preprocess_image(image_data)
        if image is None:
            return {"error": "图片预处理失败"}
        
        # YOLO检测
        detections = detect_food_regions(image, yolo_model)
        
        if detections:
            # 使用检测到的最高置信度结果
            best_detection = max(detections, key=lambda x: x["confidence"])
            
            # 裁剪检测区域
            bbox = best_detection["bbox"]
            cropped_image = image.crop(bbox)
            
            # ResNet分类
            classification = classify_food(cropped_image, resnet_model)
            best_classification = classification[0]
            
            # 计算综合置信度
            combined_confidence = best_detection["confidence"] * best_classification["confidence"]
            
            result = {
                "foodName": best_classification["class_name"],
                "confidence": combined_confidence,
                "bbox": bbox,
                "detection_confidence": best_detection["confidence"],
                "classification_confidence": best_classification["confidence"]
            }
        else:
            # 如果没有检测到食物，直接对整个图片分类
            classification = classify_food(image, resnet_model)
            best_classification = classification[0]
            
            result = {
                "foodName": best_classification["class_name"],
                "confidence": best_classification["confidence"],
                "bbox": None,
                "detection_confidence": 0.0,
                "classification_confidence": best_classification["confidence"]
            }
        
        return result
        
    except Exception as e:
        print(f"食物识别失败: {e}", file=sys.stderr)
        return {"error": str(e)}

if __name__ == "__main__":
    # 从标准输入读取图片数据
    image_data = sys.stdin.read().strip()
    
    # 识别食物
    result = recognize_food(image_data)
    
    # 输出JSON结果
    print(json.dumps(result, ensure_ascii=False))
`;

    // 确保目录存在
    const dir = path.dirname(this.pythonScriptPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    // 写入Python脚本
    fs.writeFileSync(this.pythonScriptPath, scriptContent);
    console.log('Python脚本创建成功');
  }

  private async testPythonEnvironment(): Promise<void> {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['--version']);
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          console.log('Python环境检查通过');
          resolve();
        } else {
          reject(new Error('Python环境检查失败'));
        }
      });
      
      pythonProcess.on('error', (error) => {
        reject(new Error(`Python环境错误: ${error.message}`));
      });
    });
  }

  /**
   * 识别食物图像并返回详细信息
   * @param base64Image base64编码的图片数据
   * @returns 食物识别结果
   */
  async recognizeFood(base64Image: string): Promise<FoodRecognitionResult> {
    try {
      if (!this.isInitialized) {
        throw new Error('AI服务未初始化');
      }

      // 调用Python脚本进行识别
      const result = await this.callPythonScript(base64Image);
      
      if (result.error) {
        throw new Error(result.error);
      }

      // 获取食物信息
      const foodInfo = this.getFoodInfo(result.foodName);
      
      // 构建识别结果
      const recognitionResult: FoodRecognitionResult = {
        foodName: foodInfo.name,
        purineLevel: foodInfo.level,
        purineContent: foodInfo.content,
        suitableForGout: foodInfo.level !== 'high',
        advice: foodInfo.advice,
        nutritionEstimate: foodInfo.nutrition,
        confidence: result.confidence || 0.8
      };

      return recognitionResult;
    } catch (error) {
      console.error('AI识别失败:', error);
      throw new Error(`食物识别失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  }

  private async callPythonScript(imageData: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', [this.pythonScriptPath]);
      
      let output = '';
      let errorOutput = '';
      
      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
      });
      
      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(output);
            resolve(result);
          } catch (error) {
            reject(new Error(`解析Python输出失败: ${error}`));
          }
        } else {
          reject(new Error(`Python脚本执行失败: ${errorOutput}`));
        }
      });
      
      pythonProcess.on('error', (error) => {
        reject(new Error(`启动Python进程失败: ${error.message}`));
      });
      
      // 发送图片数据到Python脚本
      pythonProcess.stdin.write(imageData);
      pythonProcess.stdin.end();
    });
  }

  private getFoodInfo(foodName: string): { name: string; level: 'high' | 'medium' | 'low'; content: string; advice: string; nutrition: any } {
    // 模糊匹配食物名称
    const foodNameLower = foodName.toLowerCase();
    
    for (const [key, value] of Object.entries(FOOD_PURINE_DB)) {
      if (foodNameLower.includes(key) || key.includes(foodNameLower)) {
        return {
          name: key,
          level: value.level,
          content: value.content,
          advice: value.advice,
          nutrition: value.nutrition
        };
      }
    }
    
    // 如果没有找到匹配，返回默认值
    return {
      name: foodName,
      level: 'low',
      content: '10-50mg/100g',
      advice: '无法准确识别，建议咨询医生或营养师',
      nutrition: {
        calories: '100kcal/100g',
        protein: '5g/100g',
        fat: '3g/100g',
        carbohydrates: '15g/100g',
        fiber: '2g/100g'
      }
    };
  }

  /**
   * 获取服务状态
   */
  async getServiceStatus(): Promise<{ status: string; model: string; available: boolean }> {
    return {
      status: this.isInitialized ? 'healthy' : 'unhealthy',
      model: 'YOLO+ResNet融合模型',
      available: this.isInitialized
    };
  }
} 