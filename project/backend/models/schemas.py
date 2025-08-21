from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class PurineLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class NutritionEstimate(BaseModel):
    calories: str = Field(..., description="热量估算，如：120kcal/100g")
    protein: str = Field(..., description="蛋白质含量，如：8g/100g")
    fat: str = Field(..., description="脂肪含量，如：2g/100g")
    carbohydrates: str = Field(..., description="碳水化合物含量，如：15g/100g")
    fiber: str = Field(..., description="膳食纤维含量，如：3g/100g")

class FoodRecognitionResult(BaseModel):
    foodName: str = Field(..., description="识别出的食物名称")
    purineLevel: PurineLevel = Field(..., description="嘌呤含量等级")
    purineContent: str = Field(..., description="具体嘌呤含量，如：150-200mg/100g")
    suitableForGout: bool = Field(..., description="是否适合痛风患者食用")
    advice: str = Field(..., description="详细的食用建议")
    nutritionEstimate: NutritionEstimate = Field(..., description="营养成分估算")
    confidence: float = Field(..., ge=0, le=1, description="识别置信度，范围0-1")
    bbox: Optional[List[float]] = Field(None, description="检测框坐标 [x1, y1, x2, y2]")

class RecognitionRequest(BaseModel):
    image: str = Field(..., description="base64编码的图片数据")

class RecognitionResponse(BaseModel):
    success: bool = Field(..., description="请求是否成功")
    data: Optional[FoodRecognitionResult] = Field(None, description="识别结果")
    error: Optional[str] = Field(None, description="错误信息")

class FoodInfo(BaseModel):
    name: str = Field(..., description="食物名称")
    purineLevel: PurineLevel = Field(..., description="嘌呤含量等级")
    purineContent: str = Field(..., description="嘌呤含量")
    suitableForGout: bool = Field(..., description="是否适合痛风患者")
    advice: str = Field(..., description="食用建议")
    nutrition: NutritionEstimate = Field(..., description="营养成分")
    category: str = Field(..., description="食物类别")

class ModelStatus(BaseModel):
    status: str = Field(..., description="模型状态")
    model_type: str = Field(..., description="模型类型")
    available: bool = Field(..., description="是否可用")
    version: str = Field(..., description="模型版本")
    last_updated: str = Field(..., description="最后更新时间")

class DetectionResult(BaseModel):
    bbox: List[float] = Field(..., description="检测框坐标")
    confidence: float = Field(..., ge=0, le=1, description="检测置信度")
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")

class ClassificationResult(BaseModel):
    food_name: str = Field(..., description="食物名称")
    confidence: float = Field(..., ge=0, le=1, description="分类置信度")
    class_id: int = Field(..., description="类别ID")
    top_k_predictions: List[Dict[str, Any]] = Field(..., description="前K个预测结果") 