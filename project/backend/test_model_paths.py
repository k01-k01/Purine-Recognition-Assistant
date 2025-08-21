#!/usr/bin/env python3
"""
测试模型路径配置
"""
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_paths():
    """测试模型路径配置"""
    print("🔍 检查模型路径配置...")
    print("=" * 50)
    
    # 检查训练好的模型文件
    model_files = [
        "models/resnet_food20_classifier.pth",
        "models/yolo_food20_detector.pt",
        "models/resnet_food_classifier.pth",  # 旧路径
    ]
    
    print("📁 训练好的模型文件:")
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"  ✅ {model_file} ({size:.1f} MB)")
        else:
            print(f"  ❌ {model_file} (不存在)")
    
    print("\n🔧 ResNet分类器配置:")
    try:
        from models.resnet_classifier import ResNetClassifier
        classifier = ResNetClassifier()
        print(f"  ✅ 模型路径: {classifier.model_path}")
        print(f"  ✅ 类别数: {classifier.num_classes}")
        print(f"  ✅ 支持类别: {len(classifier.class_names)}")
        print(f"  ✅ 设备: {classifier.device}")
    except Exception as e:
        print(f"  ❌ ResNet分类器初始化失败: {e}")
    
    print("\n🔧 训练好的模型加载器:")
    try:
        from models.trained_model import load_trained_model
        model = load_trained_model()
        if model:
            print(f"  ✅ 成功加载训练好的模型")
            print(f"  ✅ 模型路径: {model.model_path}")
        else:
            print("  ⚠️ 未找到训练好的模型")
    except Exception as e:
        print(f"  ❌ 训练好的模型加载失败: {e}")
    
    print("\n🔧 食物识别服务:")
    try:
        from services.food_recognition_service import FoodRecognitionService
        service = FoodRecognitionService()
        print(f"  ✅ 服务初始化成功")
        print(f"  ✅ 使用训练好的模型: {service.use_trained_model}")
    except Exception as e:
        print(f"  ❌ 食物识别服务初始化失败: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 模型路径配置检查完成!")

if __name__ == "__main__":
    test_model_paths() 