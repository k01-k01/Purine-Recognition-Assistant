import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from models.yolo_detector import YOLODetector
from models.resnet_classifier import ResNetClassifier
from models.trained_model import load_trained_model, TrainedYOLOResNetModel
from models.schemas import FoodRecognitionResult, PurineLevel, NutritionEstimate
from utils.food_database import FoodDatabase
from PIL import Image
import logging
import time

logger = logging.getLogger(__name__)

class FoodRecognitionService:
    """食物识别服务，整合YOLO检测器和ResNet分类器，支持训练好的模型"""
    
    def __init__(self):
        self.yolo_detector = None
        self.resnet_classifier = None
        self.trained_model = None
        self.food_database = FoodDatabase()
        self.use_trained_model = False
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        try:
            logger.info("🚀 开始初始化AI模型...")
            
            # 首先尝试加载训练好的模型
            logger.info("🔍 尝试加载训练好的模型...")
            self.trained_model = load_trained_model()
            
            if self.trained_model is not None:
                self.use_trained_model = True
                logger.info("✅ 成功加载训练好的模型，将使用训练好的模型进行识别")
                return
            
            logger.info("⚠️ 未找到训练好的模型，使用预训练模型...")
            
            # 初始化YOLO检测器
            logger.info("📦 正在加载YOLO检测器...")
            self.yolo_detector = YOLODetector()
            logger.info("✅ YOLO检测器初始化成功")
            
            # 初始化ResNet分类器
            logger.info("📦 正在加载ResNet分类器...")
            self.resnet_classifier = ResNetClassifier()
            logger.info("✅ ResNet分类器初始化成功")
            
            logger.info("🎉 所有AI模型初始化完成！")
            
        except Exception as e:
            logger.error(f"❌ 模型初始化失败: {e}")
            raise
    
    async def recognize_food(self, image_data: str) -> FoodRecognitionResult:
        """识别食物并返回详细信息"""
        start_time = time.time()
        logger.info("🔍 开始食物识别流程...")
        
        try:
            # 如果使用训练好的模型，直接进行预测
            if self.use_trained_model and self.trained_model is not None:
                logger.info("🎯 使用训练好的模型进行识别...")
                return await self._recognize_with_trained_model(image_data, start_time)
            
            # 使用预训练模型进行识别
            logger.info("🔍 使用预训练模型进行识别...")
            
            # 第一步：使用YOLO检测食物区域
            logger.info("🔍 步骤1: 使用YOLO模型检测食物区域...")
            yolo_start = time.time()
            detections = self.yolo_detector.detect(image_data)
            yolo_time = time.time() - yolo_start
            
            logger.info(f"📊 YOLO检测结果: 发现 {len(detections)} 个检测区域")
            if detections:
                for i, det in enumerate(detections[:3]):
                    logger.info(f"  检测区域 {i+1}: {det['class_name']} (置信度: {det['confidence']:.3f})")
            
            if not detections:
                # 如果没有检测到食物，直接对整个图片进行分类
                logger.info("⚠️ 未检测到食物区域，使用ResNet对整个图片进行分类")
                resnet_start = time.time()
                classification_result = self.resnet_classifier.classify(image_data)
                resnet_time = time.time() - resnet_start
                
                logger.info(f"📊 ResNet分类结果: {classification_result['food_name']} (置信度: {classification_result['confidence']:.3f})")
                logger.info(f"⏱️ ResNet分类耗时: {resnet_time:.3f}秒")
                
                total_time = time.time() - start_time
                logger.info(f"🎯 识别完成! 总耗时: {total_time:.3f}秒")
                
                return self._create_result_from_classification(classification_result)
            
            # 第二步：对检测到的区域进行分类
            logger.info("🔍 步骤2: 使用ResNet对检测区域进行分类...")
            best_result = None
            best_confidence = 0.0
            
            for i, detection in enumerate(detections[:3]):  # 只处理前3个检测结果
                try:
                    logger.info(f"🔍 处理检测区域 {i+1}: {detection['class_name']}")
                    
                    # 裁剪检测区域
                    crop_start = time.time()
                    cropped_image = self.yolo_detector.crop_detection(
                        image_data, 
                        detection["bbox"]
                    )
                    crop_time = time.time() - crop_start
                    logger.info(f"✂️ 区域裁剪耗时: {crop_time:.3f}秒")
                    
                    # 对裁剪区域进行分类
                    resnet_start = time.time()
                    classification_result = self.resnet_classifier.classify(cropped_image)
                    resnet_time = time.time() - resnet_start
                    
                    # 计算综合置信度（检测置信度 * 分类置信度）
                    combined_confidence = detection["confidence"] * classification_result["confidence"]
                    
                    logger.info(f"📊 区域 {i+1} 分类结果:")
                    logger.info(f"  - YOLO检测: {detection['class_name']} (置信度: {detection['confidence']:.3f})")
                    logger.info(f"  - ResNet分类: {classification_result['food_name']} (置信度: {classification_result['confidence']:.3f})")
                    logger.info(f"  - 综合置信度: {combined_confidence:.3f}")
                    logger.info(f"  - ResNet分类耗时: {resnet_time:.3f}秒")
                    
                    if combined_confidence > best_confidence:
                        best_confidence = combined_confidence
                        best_result = {
                            "detection": detection,
                            "classification": classification_result,
                            "combined_confidence": combined_confidence
                        }
                        logger.info(f"🏆 更新最佳结果: {classification_result['food_name']}")
                        
                except Exception as e:
                    logger.warning(f"❌ 处理检测区域 {i+1} 失败: {e}")
                    continue
            
            if best_result:
                total_time = time.time() - start_time
                logger.info(f"🎯 识别完成! 最佳结果: {best_result['classification']['food_name']}")
                logger.info(f"📊 最终置信度: {best_result['combined_confidence']:.3f}")
                logger.info(f"⏱️ 总耗时: {total_time:.3f}秒")
                logger.info(f"📈 性能统计:")
                logger.info(f"  - YOLO检测: {yolo_time:.3f}秒")
                logger.info(f"  - ResNet分类: {resnet_time:.3f}秒")
                logger.info(f"  - 总耗时: {total_time:.3f}秒")
                
                return self._create_result_from_detection_and_classification(best_result)
            else:
                # 如果所有检测区域都处理失败，对整个图片进行分类
                logger.info("⚠️ 检测区域处理失败，使用ResNet对整个图片进行分类")
                resnet_start = time.time()
                classification_result = self.resnet_classifier.classify(image_data)
                resnet_time = time.time() - resnet_start
                
                logger.info(f"📊 ResNet分类结果: {classification_result['food_name']} (置信度: {classification_result['confidence']:.3f})")
                logger.info(f"⏱️ ResNet分类耗时: {resnet_time:.3f}秒")
                
                total_time = time.time() - start_time
                logger.info(f"🎯 识别完成! 总耗时: {total_time:.3f}秒")
                
                return self._create_result_from_classification(classification_result)
                
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ 食物识别失败 (耗时: {total_time:.3f}秒): {e}")
            raise
    
    def _create_result_from_classification(self, classification_result: Dict[str, Any]) -> FoodRecognitionResult:
        """从分类结果创建识别结果"""
        logger.info(f"📝 创建识别结果: {classification_result['food_name']}")
        return FoodRecognitionResult(
            foodName=classification_result["food_name"],
            purineLevel=PurineLevel(classification_result["purine_level"]),
            purineContent=classification_result["purine_content"],
            suitableForGout=classification_result["suitable_for_gout"],
            advice=classification_result["advice"],
            nutritionEstimate=NutritionEstimate(**classification_result["nutrition"]),
            confidence=classification_result["confidence"],
            bbox=None
        )
    
    def _create_result_from_detection_and_classification(self, result: Dict[str, Any]) -> FoodRecognitionResult:
        """从检测和分类结果创建识别结果"""
        detection = result["detection"]
        classification = result["classification"]
        
        logger.info(f"📝 创建融合识别结果: {classification['food_name']}")
        logger.info(f"  - 检测框: {detection['bbox']}")
        logger.info(f"  - 综合置信度: {result['combined_confidence']:.3f}")
        
        return FoodRecognitionResult(
            foodName=classification["food_name"],
            purineLevel=PurineLevel(classification["purine_level"]),
            purineContent=classification["purine_content"],
            suitableForGout=classification["suitable_for_gout"],
            advice=classification["advice"],
            nutritionEstimate=NutritionEstimate(**classification["nutrition"]),
            confidence=result["combined_confidence"],
            bbox=detection["bbox"]
        )
    
    async def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        if self.use_trained_model and self.trained_model is not None:
            # 使用训练好的模型
            logger.info(f"📊 服务状态检查 (训练好的模型):")
            logger.info(f"  - 训练好的模型: ✅ 就绪")
            
            model_info = self.trained_model.get_model_info()
            return {
                "status": "healthy",
                "model_type": "训练好的YOLO+ResNet融合模型",
                "available": True,
                "version": "3.0.0",
                "last_updated": "2024-01-01T00:00:00Z",
                "models": {
                    "trained_model": {
                        "ready": True,
                        "info": model_info
                    }
                }
            }
        else:
            # 使用预训练模型
            yolo_ready = self.yolo_detector.is_ready() if self.yolo_detector else False
            resnet_ready = self.resnet_classifier.is_ready() if self.resnet_classifier else False
            
            logger.info(f"📊 服务状态检查 (预训练模型):")
            logger.info(f"  - YOLO检测器: {'✅ 就绪' if yolo_ready else '❌ 未就绪'}")
            logger.info(f"  - ResNet分类器: {'✅ 就绪' if resnet_ready else '❌ 未就绪'}")
            
            return {
                "status": "healthy" if (yolo_ready and resnet_ready) else "unhealthy",
                "model_type": "预训练YOLO+ResNet融合模型",
                "available": yolo_ready and resnet_ready,
                "version": "2.0.0",
                "last_updated": "2024-01-01T00:00:00Z",
                "models": {
                    "yolo_detector": {
                        "ready": yolo_ready,
                        "info": self.yolo_detector.get_model_info() if self.yolo_detector else None
                    },
                    "resnet_classifier": {
                        "ready": resnet_ready,
                        "info": self.resnet_classifier.get_model_info() if self.resnet_classifier else None
                    }
                }
            }
    
    def get_supported_foods(self) -> List[str]:
        """获取支持的食物类别"""
        if self.resnet_classifier:
            foods = self.resnet_classifier.get_supported_classes()
            logger.info(f"📋 支持的食物类别数量: {len(foods)}")
            return foods
        return []
    
    async def batch_recognize(self, image_data_list: List[str]) -> List[FoodRecognitionResult]:
        """批量识别食物"""
        logger.info(f"🔄 开始批量识别，图片数量: {len(image_data_list)}")
        results = []
        
        for i, image_data in enumerate(image_data_list):
            try:
                logger.info(f"🔍 处理图片 {i+1}/{len(image_data_list)}")
                result = await self.recognize_food(image_data)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ 批量识别失败 (图片 {i+1}): {e}")
                # 创建一个默认结果
                default_result = FoodRecognitionResult(
                    foodName="未知食物",
                    purineLevel=PurineLevel.LOW,
                    purineContent="10-50mg/100g",
                    suitableForGout=True,
                    advice="无法识别，建议咨询医生",
                    nutritionEstimate=NutritionEstimate(
                        calories="未知",
                        protein="未知",
                        fat="未知",
                        carbohydrates="未知",
                        fiber="未知"
                    ),
                    confidence=0.0
                )
                results.append(default_result)
        
        logger.info(f"✅ 批量识别完成，成功处理: {len(results)} 张图片")
        return results
    
    async def _recognize_with_trained_model(self, image_data: str, start_time: float) -> FoodRecognitionResult:
        """使用训练好的模型进行识别"""
        try:
            # 解码图像
            import base64
            import io
            
            # 处理base64数据，移除可能的data URL前缀
            if image_data.startswith('data:image'):
                # 处理data URL格式
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            else:
                # 直接处理base64字符串
                image_bytes = base64.b64decode(image_data)
            
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # 使用训练好的模型进行预测
            prediction_result = self.trained_model.predict(image)
            
            # 获取预测结果
            food_name = prediction_result.get('food_name', 'unknown')
            confidence = prediction_result.get('confidence', 0.0)
            
            # 获取食物信息
            food_info = self.food_database.get_food_info(food_name)
            if food_info is None:
                # 如果找不到食物信息，使用默认值
                food_info = {
                    'purine_level': 'LOW',
                    'purine_content': '10-50mg/100g',
                    'suitable_for_gout': True,
                    'dietary_advice': '无法获取详细信息，建议咨询医生',
                    'nutrition': {
                        'calories': '未知',
                        'protein': '未知',
                        'fat': '未知',
                        'carbohydrates': '未知',
                        'fiber': '未知'
                    }
                }
            else:
                # 转换为字典格式
                food_info = {
                    'purine_level': food_info.purineLevel.value,
                    'purine_content': food_info.purineContent,
                    'suitable_for_gout': food_info.suitableForGout,
                    'dietary_advice': food_info.advice,
                    'nutrition': {
                        'calories': food_info.nutrition.calories,
                        'protein': food_info.nutrition.protein,
                        'fat': food_info.nutrition.fat,
                        'carbohydrates': food_info.nutrition.carbohydrates,
                        'fiber': food_info.nutrition.fiber
                    }
                }
            
            total_time = time.time() - start_time
            logger.info(f"🎯 训练模型识别完成! 总耗时: {total_time:.3f}秒")
            logger.info(f"📊 识别结果: {food_name} (置信度: {confidence:.3f})")
            
            return FoodRecognitionResult(
                foodName=food_name,
                purineLevel=PurineLevel(food_info.get('purine_level', 'LOW')),
                purineContent=food_info.get('purine_content', '10-50mg/100g'),
                suitableForGout=food_info.get('suitable_for_gout', True),
                advice=food_info.get('dietary_advice', ''),
                nutritionEstimate=NutritionEstimate(
                    calories=food_info.get('nutrition', {}).get('calories', '未知'),
                    protein=food_info.get('nutrition', {}).get('protein', '未知'),
                    fat=food_info.get('nutrition', {}).get('fat', '未知'),
                    carbohydrates=food_info.get('nutrition', {}).get('carbohydrates', '未知'),
                    fiber=food_info.get('nutrition', {}).get('fiber', '未知')
                ),
                confidence=confidence,
                bbox=None
            )
            
        except Exception as e:
            logger.error(f"❌ 训练模型识别失败: {e}")
            # 如果训练模型失败，返回默认结果
            return FoodRecognitionResult(
                foodName="unknown",
                purineLevel=PurineLevel.LOW,
                purineContent="10-50mg/100g",
                suitableForGout=True,
                advice="无法识别食物，请重新上传图片",
                nutritionEstimate=NutritionEstimate(
                    calories="未知",
                    protein="未知",
                    fat="未知",
                    carbohydrates="未知",
                    fiber="未知"
                ),
                confidence=0.0,
                bbox=None
            ) 