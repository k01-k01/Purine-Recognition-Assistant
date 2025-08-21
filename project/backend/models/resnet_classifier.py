import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import base64
import io
import os
import json
from typing import Dict, List, Any, Optional
from utils.food_database import FoodDatabase
import logging

logger = logging.getLogger(__name__)

class ResNetClassifier:
    """ResNet食物分类器"""
    
    def __init__(self, model_path: str = "models/resnet_food20_classifier.pth", num_classes: int = 20):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes  # 动态设置类别数
        self.food_database = FoodDatabase()
        self.class_names = self._load_class_names()
        self.transform = self._get_transforms()
        
        logger.info(f"🔧 ResNet分类器配置:")
        logger.info(f"  - 设备: {self.device}")
        logger.info(f"  - 类别数: {self.num_classes}")
        logger.info(f"  - 支持的食物类别: {len(self.class_names)}")
        
        self._load_model()
    
    def _load_class_names(self) -> List[str]:
        """加载类别名称"""
        if self.num_classes == 20:
            # Food20数据集的类别名称（与实际数据集匹配）
            food20_classes = [
                "Apple", "Banana", "Bean", "Bottle_Gourd", "Broccoli", "Cabbage", 
                "Capsicum", "Carrot", "Cauliflower", "Cucumber", "Grape", 
                "Grapefruit", "Mango", "Meat", "Pear", "Pineapple", 
                "Potato", "Pumpkin", "Radish", "Tomato"
            ]
            return food20_classes
        elif self.num_classes == 101:
            # Food-101数据集的类别名称
            food101_classes = [
                "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
                "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
                "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
                "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
                "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
                "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
                "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
                "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
                "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
                "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon",
                "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros",
                "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich",
                "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette",
                "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta",
                "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
                "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops",
                "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
                "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki",
                "tiramisu", "tuna_tartare", "waffles"
            ]
            return food101_classes
        else:
            # 通用类别名称（数字编号）
            return [f"class_{i}" for i in range(self.num_classes)]
    
    def _get_transforms(self):
        """获取图像预处理变换"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self):
        """加载ResNet模型"""
        try:
            logger.info("📦 开始加载ResNet模型...")
            
            # 创建ResNet50模型
            logger.info("🏗️ 创建ResNet50模型架构...")
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            
            # 修改最后一层以匹配我们的类别数
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
            logger.info(f"🔧 修改最后一层: {num_ftrs} -> {self.num_classes}")
            
            # 如果存在训练好的模型权重，加载它
            if os.path.exists(self.model_path):
                logger.info(f"📁 加载训练好的ResNet模型: {self.model_path}")
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            else:
                logger.info("📁 使用预训练的ResNet50模型")
            
            # 将模型移动到指定设备并设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ ResNet模型已加载到设备: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ 加载ResNet模型失败: {e}")
            raise
    
    def preprocess_image(self, image_data: str) -> torch.Tensor:
        """预处理图片数据"""
        try:
            logger.debug("🖼️ 开始预处理图片...")
            
            # 解码base64图片
            if image_data.startswith('data:image'):
                # 处理data URL格式
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            else:
                # 直接处理base64字符串
                image_bytes = base64.b64decode(image_data)
            
            # 转换为PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            logger.debug(f"📐 原始图片尺寸: {image.size}")
            
            # 应用预处理变换
            image_tensor = self.transform(image).unsqueeze(0)
            logger.debug(f"✅ 图片预处理完成，张量形状: {image_tensor.shape}")
            
            return image_tensor
            
        except Exception as e:
            logger.error(f"❌ 图片预处理失败: {e}")
            raise
    
    def classify(self, image_data: str, top_k: int = 5) -> Dict[str, Any]:
        """分类图片中的食物"""
        try:
            logger.info("🔍 开始ResNet分类...")
            
            # 预处理图片
            image_tensor = self.preprocess_image(image_data)
            image_tensor = image_tensor.to(self.device)
            
            # 进行预测
            logger.debug("🤖 运行ResNet推理...")
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 获取top-k预测结果
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                # 转换为列表
                top_probs = top_probs.cpu().numpy()[0]
                top_indices = top_indices.cpu().numpy()[0]
                
                logger.info(f"📊 ResNet推理完成，获取前{top_k}个预测结果")
                
                # 构建预测结果
                predictions = []
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    class_name = self.class_names[idx]
                    # 将下划线替换为空格，使名称更易读
                    readable_name = class_name.replace('_', ' ').title()
                    
                    predictions.append({
                        "class_name": readable_name,
                        "original_name": class_name,
                        "confidence": float(prob),
                        "class_id": int(idx)
                    })
                    
                    logger.debug(f"  预测 {i+1}: {readable_name} (置信度: {prob:.3f})")
                
                # 获取最佳预测
                best_prediction = predictions[0]
                logger.info(f"🏆 最佳预测: {best_prediction['class_name']} (置信度: {best_prediction['confidence']:.3f})")
                
                # 从食物数据库获取详细信息
                food_info = self.food_database.get_food_info(best_prediction["class_name"])
                
                result = {
                    "food_name": food_info.name,
                    "confidence": best_prediction["confidence"],
                    "class_id": best_prediction["class_id"],
                    "top_k_predictions": predictions,
                    "purine_level": food_info.purineLevel.value,
                    "purine_content": food_info.purineContent,
                    "suitable_for_gout": food_info.suitableForGout,
                    "advice": food_info.advice,
                    "nutrition": {
                        "calories": food_info.nutrition.calories,
                        "protein": food_info.nutrition.protein,
                        "fat": food_info.nutrition.fat,
                        "carbohydrates": food_info.nutrition.carbohydrates,
                        "fiber": food_info.nutrition.fiber
                    }
                }
                
                logger.info(f"✅ ResNet分类完成: {result['food_name']}")
                return result
                
        except Exception as e:
            logger.error(f"❌ ResNet分类失败: {e}")
            raise
    
    def get_supported_classes(self) -> List[str]:
        """获取支持的类别列表"""
        return [name.replace('_', ' ').title() for name in self.class_names]
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": "ResNet50",
            "model_path": self.model_path,
            "device": self.device,
            "num_classes": self.num_classes,
            "available": self.model is not None
        }
    
    def is_ready(self) -> bool:
        """检查模型是否准备就绪"""
        return self.model is not None 