import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import torch
import os
from PIL import Image
import base64
import io
import logging

logger = logging.getLogger(__name__)

class YOLODetector:
    """YOLO食物检测器"""
    
    def __init__(self, model_path: str = "models/yolo_food_detector.pt"):
        self.model_path = model_path
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        logger.info(f"🔧 YOLO检测器配置:")
        logger.info(f"  - 设备: {self.device}")
        logger.info(f"  - 置信度阈值: {self.confidence_threshold}")
        logger.info(f"  - IoU阈值: {self.iou_threshold}")
        self._load_model()
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            logger.info("📦 开始加载YOLO模型...")
            
            # 如果自定义模型不存在，使用预训练模型
            if os.path.exists(self.model_path):
                logger.info(f"📁 加载自定义YOLO模型: {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                # 使用YOLOv8n预训练模型作为基础
                logger.info("📁 使用YOLOv8n预训练模型")
                self.model = YOLO("yolov8n.pt")
            
            # 将模型移动到指定设备
            self.model.to(self.device)
            logger.info(f"✅ YOLO模型已加载到设备: {self.device}")
            
        except Exception as e:
            logger.error(f"❌ 加载YOLO模型失败: {e}")
            raise
    
    def preprocess_image(self, image_data: str) -> np.ndarray:
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
            image = Image.open(io.BytesIO(image_bytes))
            
            # 转换为OpenCV格式
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            logger.debug(f"✅ 图片预处理完成，尺寸: {image_cv.shape}")
            return image_cv
            
        except Exception as e:
            logger.error(f"❌ 图片预处理失败: {e}")
            raise
    
    def detect(self, image_data: str) -> List[Dict[str, Any]]:
        """检测图片中的食物"""
        try:
            logger.info("🔍 开始YOLO检测...")
            
            # 预处理图片
            image = self.preprocess_image(image_data)
            
            # 使用YOLO进行检测
            logger.debug("🤖 运行YOLO推理...")
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    logger.info(f"📊 检测到 {len(boxes)} 个目标")
                    
                    for i, box in enumerate(boxes):
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 获取置信度
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # 获取类别ID和名称
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        logger.debug(f"  目标 {i+1}: {class_name} (置信度: {confidence:.3f})")
                        
                        # 只保留食物相关的检测结果
                        if self._is_food_class(class_name):
                            detections.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": confidence,
                                "class_id": class_id,
                                "class_name": class_name
                            })
                            logger.info(f"🍽️ 食物检测: {class_name} (置信度: {confidence:.3f})")
                        else:
                            logger.debug(f"⏭️ 跳过非食物类别: {class_name}")
            
            # 按置信度排序
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"✅ YOLO检测完成，发现 {len(detections)} 个食物目标")
            return detections
            
        except Exception as e:
            logger.error(f"❌ YOLO检测失败: {e}")
            raise
    
    def _is_food_class(self, class_name: str) -> bool:
        """判断是否为食物类别"""
        # COCO数据集的80个类别中，食物相关的类别
        food_classes = [
            "person",  # 可能包含食物
            "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
        
        # 特别关注食物相关的类别
        food_keywords = [
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
            "hot dog", "pizza", "donut", "cake", "bottle", "wine glass", 
            "cup", "fork", "knife", "spoon", "bowl", "dining table"
        ]
        
        return class_name.lower() in [f.lower() for f in food_keywords]
    
    def crop_detection(self, image_data: str, bbox: List[float]) -> str:
        """裁剪检测区域并返回base64编码"""
        try:
            logger.debug("✂️ 开始裁剪检测区域...")
            
            # 预处理图片
            image = self.preprocess_image(image_data)
            
            # 裁剪检测区域
            x1, y1, x2, y2 = map(int, bbox)
            cropped = image[y1:y2, x1:x2]
            
            # 转换为PIL Image
            cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            
            # 转换为base64
            buffer = io.BytesIO()
            cropped_pil.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            logger.debug(f"✅ 区域裁剪完成，尺寸: {cropped.shape}")
            return img_str
            
        except Exception as e:
            logger.error(f"❌ 裁剪检测区域失败: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_type": "YOLO",
            "model_path": self.model_path,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "available": self.model is not None
        }
    
    def is_ready(self) -> bool:
        """检查模型是否准备就绪"""
        return self.model is not None 