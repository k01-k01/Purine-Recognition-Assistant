#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练好的YOLO+ResNet融合模型加载和推理
用于后端服务集成
"""

import os
import json
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class TrainedYOLOResNetModel:
    """训练好的YOLO+ResNet融合模型"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        初始化训练好的模型
        
        Args:
            model_path: 模型文件路径
            device: 设备 ('cpu', 'cuda', 'auto')
        """
        self.model_path = Path(model_path)
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载模型配置和权重
        self.model = None
        self.classes = []
        self.num_classes = 0
        self.transform = None
        
        self._load_model()
        self._setup_transform()
        
        logger.info(f"✅ 训练好的模型加载完成: {len(self.classes)} 个类别")
        logger.info(f"📋 类别列表: {self.classes}")
    
    def _load_model(self):
        """加载模型"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        logger.info(f"📂 加载模型: {self.model_path}")
        
        # 加载模型权重
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        # 检查是否是标准的state_dict格式
        if isinstance(state_dict, dict) and 'fc.weight' in state_dict:
            # 这是标准的ResNet state_dict格式
            logger.info("📋 检测到标准ResNet state_dict格式")
            
            # 从最后一层推断类别数
            self.num_classes = state_dict['fc.weight'].shape[0]
            logger.info(f"🔧 从模型推断类别数: {self.num_classes}")
            
            # 设置Food20类别名称
            if self.num_classes == 20:
                self.classes = [
                    "Apple", "Banana", "Bean", "Bottle_Gourd", "Broccoli", "Cabbage", 
                    "Carrot", "Cauliflower", "Cucumber", "Grapes", "Jalapeno", "Kiwi", 
                    "Lemon", "Lettuce", "Mango", "Onion", "Orange", "Paprika", "Pear", "Pineapple"
                ]
            else:
                self.classes = [f"class_{i}" for i in range(self.num_classes)]
            
            # 创建ResNet50模型
            from torchvision.models import resnet50
            import torch.nn as nn
            
            self.model = resnet50(weights=None)  # 不使用预训练权重
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            
            # 加载权重
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"🔧 模型架构: ResNet50")
            logger.info(f"📊 模型参数: {sum(p.numel() for p in self.model.parameters()):,} 个参数")
            
        else:
            # 这是检查点格式，包含额外的元数据
            logger.info("📋 检测到检查点格式")
            
            # 获取配置信息
            self.classes = state_dict.get('classes', [])
            self.num_classes = state_dict.get('num_classes', len(self.classes))
            config = state_dict.get('config', {})
            
            # 创建模型架构
            from train_yolo_resnet import YOLOResNetFusion
            self.model = YOLOResNetFusion(
                num_classes=self.num_classes,
                yolo_model_path=config.get('yolo_model_path')
            ).to(self.device)
            
            # 加载权重
            self.model.load_state_dict(state_dict['model_state_dict'])
            self.model.eval()
            
            logger.info(f"🔧 模型架构: YOLO+ResNet50融合")
            logger.info(f"📊 模型参数: {sum(p.numel() for p in self.model.parameters()):,} 个参数")
    
    def _setup_transform(self):
        """设置图像预处理"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image: Image.Image) -> Dict:
        """
        预测图像
        
        Args:
            image: PIL图像对象
        
        Returns:
            预测结果字典
        """
        try:
            # 预处理图像
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 获取预测结果
            predicted_food = self.classes[predicted_class]
            
            # 获取top-3预测结果
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)
            top3_results = []
            for prob, idx in zip(top3_probs, top3_indices):
                top3_results.append({
                    'food_name': self.classes[idx.item()],
                    'confidence': prob.item()
                })
            
            result = {
                'food_name': predicted_food,
                'confidence': confidence,
                'top3_predictions': top3_results,
                'model_type': 'trained_yolo_resnet_fusion'
            }
            
            logger.info(f"🔍 预测结果: {predicted_food} (置信度: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 预测失败: {e}")
            return {
                'food_name': 'unknown',
                'confidence': 0.0,
                'top3_predictions': [],
                'model_type': 'trained_yolo_resnet_fusion',
                'error': str(e)
            }
    
    def predict_batch(self, images: List[Image.Image]) -> List[Dict]:
        """
        批量预测
        
        Args:
            images: PIL图像对象列表
        
        Returns:
            预测结果列表
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_type': 'trained_yolo_resnet_fusion',
            'num_classes': self.num_classes,
            'classes': self.classes,
            'device': str(self.device),
            'model_path': str(self.model_path)
        }

def load_trained_model(model_path: str = None) -> Optional[TrainedYOLOResNetModel]:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型路径，如果为None则自动查找
    
    Returns:
        训练好的模型实例
    """
    if model_path is None:
        # 自动查找训练好的模型
        possible_paths = [
            Path('models/resnet_food20_classifier.pth'),  # Food20训练好的ResNet模型
            Path('models/yolo_food20_detector.pt'),       # Food20训练好的YOLO模型
            Path('outputs/training/models/best_model.pth'),  # 其他可能的路径
            Path('outputs/training/models/final_model.pth')
        ]
        
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                logger.info(f"🔍 找到训练好的模型: {model_path}")
                break
    
    if model_path is None:
        logger.warning("⚠️ 未找到训练好的模型文件")
        return None
    
    try:
        model = TrainedYOLOResNetModel(model_path)
        logger.info(f"✅ 成功加载训练好的模型: {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ 加载训练好的模型失败: {e}")
        return None

# 测试函数
def test_trained_model():
    """测试训练好的模型"""
    # 创建测试图像
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # 加载模型
    model = load_trained_model()
    if model is None:
        print("❌ 无法加载训练好的模型")
        return
    
    # 测试预测
    result = model.predict(test_image)
    print(f"测试预测结果: {result}")
    
    # 显示模型信息
    info = model.get_model_info()
    print(f"模型信息: {info}")

if __name__ == '__main__':
    test_trained_model() 