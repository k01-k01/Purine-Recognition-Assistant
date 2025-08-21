import json
import os
from typing import Dict, List, Optional
from models.schemas import FoodInfo, PurineLevel, NutritionEstimate

class FoodDatabase:
    """食物数据库，包含嘌呤含量和营养信息"""
    
    def __init__(self, db_path: str = "data/food_database.json"):
        self.db_path = db_path
        self.foods = self._load_food_database()
    
    def _load_food_database(self) -> Dict[str, Dict]:
        """加载食物数据库"""
        # 如果数据库文件不存在，创建默认数据库
        if not os.path.exists(self.db_path):
            self._create_default_database()
        
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载食物数据库失败: {e}")
            return self._get_default_foods()
    
    def _create_default_database(self):
        """创建默认的食物数据库"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        default_foods = self._get_default_foods()
        
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(default_foods, f, ensure_ascii=False, indent=2)
    
    def _get_default_foods(self) -> Dict[str, Dict]:
        """获取默认食物数据"""
        return {
            "apple": {
                "purineLevel": "low",
                "purineContent": "5-15mg/100g",
                "suitableForGout": True,
                "advice": "低嘌呤食物，痛风患者可以放心食用",
                "nutrition": {
                    "calories": "52kcal/100g",
                    "protein": "0.3g/100g",
                    "fat": "0.2g/100g",
                    "carbohydrates": "14g/100g",
                    "fiber": "2.4g/100g"
                },
                "category": "水果"
            },
            "banana": {
                "purineLevel": "low",
                "purineContent": "5-10mg/100g",
                "suitableForGout": True,
                "advice": "低嘌呤食物，富含钾元素，有助于尿酸排泄",
                "nutrition": {
                    "calories": "89kcal/100g",
                    "protein": "1.1g/100g",
                    "fat": "0.3g/100g",
                    "carbohydrates": "23g/100g",
                    "fiber": "2.6g/100g"
                },
                "category": "水果"
            },
            "orange": {
                "purineLevel": "low",
                "purineContent": "5-10mg/100g",
                "suitableForGout": True,
                "advice": "低嘌呤食物，富含维生素C，有助于尿酸排泄",
                "nutrition": {
                    "calories": "47kcal/100g",
                    "protein": "0.9g/100g",
                    "fat": "0.1g/100g",
                    "carbohydrates": "12g/100g",
                    "fiber": "2.4g/100g"
                },
                "category": "水果"
            },
            "carrot": {
                "purineLevel": "low",
                "purineContent": "10-20mg/100g",
                "suitableForGout": True,
                "advice": "低嘌呤食物，富含胡萝卜素，痛风患者可以放心食用",
                "nutrition": {
                    "calories": "41kcal/100g",
                    "protein": "0.9g/100g",
                    "fat": "0.2g/100g",
                    "carbohydrates": "10g/100g",
                    "fiber": "2.8g/100g"
                },
                "category": "蔬菜"
            },
            "broccoli": {
                "purineLevel": "low",
                "purineContent": "20-30mg/100g",
                "suitableForGout": True,
                "advice": "低嘌呤食物，富含维生素和矿物质，痛风患者可以放心食用",
                "nutrition": {
                    "calories": "34kcal/100g",
                    "protein": "2.8g/100g",
                    "fat": "0.4g/100g",
                    "carbohydrates": "7g/100g",
                    "fiber": "2.6g/100g"
                },
                "category": "蔬菜"
            },
            "pizza": {
                "purineLevel": "medium",
                "purineContent": "50-100mg/100g",
                "suitableForGout": True,
                "advice": "中等嘌呤食物，痛风患者可适量食用，注意控制量",
                "nutrition": {
                    "calories": "266kcal/100g",
                    "protein": "11g/100g",
                    "fat": "10g/100g",
                    "carbohydrates": "33g/100g",
                    "fiber": "2.5g/100g"
                },
                "category": "主食"
            },
            "hamburger": {
                "purineLevel": "medium",
                "purineContent": "80-120mg/100g",
                "suitableForGout": True,
                "advice": "中等嘌呤食物，痛风患者可适量食用，建议选择瘦肉",
                "nutrition": {
                    "calories": "295kcal/100g",
                    "protein": "17g/100g",
                    "fat": "12g/100g",
                    "carbohydrates": "30g/100g",
                    "fiber": "1.2g/100g"
                },
                "category": "主食"
            },
            "hot dog": {
                "purineLevel": "high",
                "purineContent": "150-200mg/100g",
                "suitableForGout": False,
                "advice": "高嘌呤食物，痛风患者应避免食用",
                "nutrition": {
                    "calories": "290kcal/100g",
                    "protein": "12g/100g",
                    "fat": "26g/100g",
                    "carbohydrates": "4g/100g",
                    "fiber": "0g/100g"
                },
                "category": "肉类"
            },
            "cake": {
                "purineLevel": "low",
                "purineContent": "10-20mg/100g",
                "suitableForGout": True,
                "advice": "低嘌呤食物，但含糖量高，痛风患者应适量食用",
                "nutrition": {
                    "calories": "257kcal/100g",
                    "protein": "4.5g/100g",
                    "fat": "12g/100g",
                    "carbohydrates": "35g/100g",
                    "fiber": "0.8g/100g"
                },
                "category": "甜点"
            },
            "donut": {
                "purineLevel": "low",
                "purineContent": "10-20mg/100g",
                "suitableForGout": True,
                "advice": "低嘌呤食物，但含糖量和脂肪含量高，应适量食用",
                "nutrition": {
                    "calories": "253kcal/100g",
                    "protein": "4.3g/100g",
                    "fat": "14g/100g",
                    "carbohydrates": "31g/100g",
                    "fiber": "1.2g/100g"
                },
                "category": "甜点"
            },
            "sandwich": {
                "purineLevel": "medium",
                "purineContent": "50-100mg/100g",
                "suitableForGout": True,
                "advice": "中等嘌呤食物，痛风患者可适量食用",
                "nutrition": {
                    "calories": "250kcal/100g",
                    "protein": "15g/100g",
                    "fat": "8g/100g",
                    "carbohydrates": "30g/100g",
                    "fiber": "2g/100g"
                },
                "category": "主食"
            },
            "person": {
                "purineLevel": "low",
                "purineContent": "0mg/100g",
                "suitableForGout": True,
                "advice": "这不是食物，请上传食物图片",
                "nutrition": {
                    "calories": "0kcal/100g",
                    "protein": "0g/100g",
                    "fat": "0g/100g",
                    "carbohydrates": "0g/100g",
                    "fiber": "0g/100g"
                },
                "category": "其他"
            },
            "unknown": {
                "purineLevel": "low",
                "purineContent": "10-50mg/100g",
                "suitableForGout": True,
                "advice": "无法准确识别，建议咨询医生或营养师",
                "nutrition": {
                    "calories": "100kcal/100g",
                    "protein": "5g/100g",
                    "fat": "3g/100g",
                    "carbohydrates": "15g/100g",
                    "fiber": "2g/100g"
                },
                "category": "其他"
            }
        }
    
    def get_food_info(self, food_name: str) -> Optional[FoodInfo]:
        """获取特定食物的信息"""
        # 模糊匹配
        for key, value in self.foods.items():
            if food_name.lower() in key.lower() or key.lower() in food_name.lower():
                return FoodInfo(
                    name=key,
                    purineLevel=PurineLevel(value["purineLevel"]),
                    purineContent=value["purineContent"],
                    suitableForGout=value["suitableForGout"],
                    advice=value["advice"],
                    nutrition=NutritionEstimate(**value["nutrition"]),
                    category=value["category"]
                )
        
        # 如果没有找到，返回默认的蔬菜信息
        return FoodInfo(
            name=food_name,
            purineLevel=PurineLevel.LOW,
            purineContent="10-50mg/100g",
            suitableForGout=True,
            advice="低嘌呤食物，痛风患者可以放心食用。",
            nutrition=NutritionEstimate(
                calories="20-50kcal/100g",
                protein="1-3g/100g",
                fat="0.1-0.5g/100g",
                carbohydrates="3-10g/100g",
                fiber="1-3g/100g"
            ),
            category="其他"
        )
    
    def get_all_foods(self) -> List[FoodInfo]:
        """获取所有食物信息"""
        return [
            FoodInfo(
                name=name,
                purineLevel=PurineLevel(data["purineLevel"]),
                purineContent=data["purineContent"],
                suitableForGout=data["suitableForGout"],
                advice=data["advice"],
                nutrition=NutritionEstimate(**data["nutrition"]),
                category=data["category"]
            )
            for name, data in self.foods.items()
        ]
    
    def search_foods(self, keyword: str) -> List[FoodInfo]:
        """搜索食物"""
        results = []
        keyword_lower = keyword.lower()
        
        for name, data in self.foods.items():
            if (keyword_lower in name.lower() or 
                keyword_lower in data["category"].lower()):
                results.append(FoodInfo(
                    name=name,
                    purineLevel=PurineLevel(data["purineLevel"]),
                    purineContent=data["purineContent"],
                    suitableForGout=data["suitableForGout"],
                    advice=data["advice"],
                    nutrition=NutritionEstimate(**data["nutrition"]),
                    category=data["category"]
                ))
        
        return results
    
    def get_foods_by_category(self, category: str) -> List[FoodInfo]:
        """按类别获取食物"""
        results = []
        category_lower = category.lower()
        
        for name, data in self.foods.items():
            if category_lower in data["category"].lower():
                results.append(FoodInfo(
                    name=name,
                    purineLevel=PurineLevel(data["purineLevel"]),
                    purineContent=data["purineContent"],
                    suitableForGout=data["suitableForGout"],
                    advice=data["advice"],
                    nutrition=NutritionEstimate(**data["nutrition"]),
                    category=data["category"]
                ))
        
        return results
    
    def get_foods_by_purine_level(self, level: PurineLevel) -> List[FoodInfo]:
        """按嘌呤等级获取食物"""
        results = []
        
        for name, data in self.foods.items():
            if PurineLevel(data["purineLevel"]) == level:
                results.append(FoodInfo(
                    name=name,
                    purineLevel=PurineLevel(data["purineLevel"]),
                    purineContent=data["purineContent"],
                    suitableForGout=data["suitableForGout"],
                    advice=data["advice"],
                    nutrition=NutritionEstimate(**data["nutrition"]),
                    category=data["category"]
                ))
        
        return results 