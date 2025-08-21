#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的YOLO+ResNet融合模型
"""

import os
import sys
import time
from pathlib import Path
from PIL import Image
import numpy as np

def test_model_loading():
    """测试模型加载"""
    print("🔍 测试模型加载...")
    
    try:
        from models.trained_model import load_trained_model
        
        model = load_trained_model()
        if model is None:
            print("❌ 未找到训练好的模型")
            return False
        
        print("✅ 模型加载成功")
        
        # 显示模型信息
        info = model.get_model_info()
        print(f"📊 模型信息:")
        print(f"  - 类型: {info['model_type']}")
        print(f"  - 类别数: {info['num_classes']}")
        print(f"  - 设备: {info['device']}")
        print(f"  - 路径: {info['model_path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def test_prediction():
    """测试模型预测"""
    print("\n🔍 测试模型预测...")
    
    try:
        from models.trained_model import load_trained_model
        
        model = load_trained_model()
        if model is None:
            print("❌ 无法加载模型进行预测测试")
            return False
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # 进行预测
        start_time = time.time()
        result = model.predict(test_image)
        prediction_time = time.time() - start_time
        
        print("✅ 预测成功")
        print(f"📊 预测结果:")
        print(f"  - 食物名称: {result['food_name']}")
        print(f"  - 置信度: {result['confidence']:.3f}")
        print(f"  - 预测时间: {prediction_time:.3f}秒")
        
        # 显示top-3预测结果
        if 'top3_predictions' in result:
            print(f"  - Top-3预测:")
            for i, pred in enumerate(result['top3_predictions'][:3]):
                print(f"    {i+1}. {pred['food_name']} ({pred['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ 预测测试失败: {e}")
        return False

def test_batch_prediction():
    """测试批量预测"""
    print("\n🔍 测试批量预测...")
    
    try:
        from models.trained_model import load_trained_model
        
        model = load_trained_model()
        if model is None:
            print("❌ 无法加载模型进行批量预测测试")
            return False
        
        # 创建多个测试图像
        test_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        
        # 进行批量预测
        start_time = time.time()
        results = model.predict_batch(test_images)
        batch_time = time.time() - start_time
        
        print("✅ 批量预测成功")
        print(f"📊 批量预测结果:")
        print(f"  - 图像数量: {len(test_images)}")
        print(f"  - 总耗时: {batch_time:.3f}秒")
        print(f"  - 平均耗时: {batch_time/len(test_images):.3f}秒/张")
        
        for i, result in enumerate(results):
            print(f"  图像 {i+1}: {result['food_name']} ({result['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量预测测试失败: {e}")
        return False

def test_with_real_image():
    """使用真实图像测试"""
    print("\n🔍 使用真实图像测试...")
    
    # 查找数据集中的图像
    dataset_path = Path("datasets/Food20_new")
    if not dataset_path.exists():
        print("❌ 数据集不存在，跳过真实图像测试")
        return False
    
    # 查找训练集中的图像
    train_images_path = dataset_path / "train" / "images"
    if not train_images_path.exists():
        print("❌ 训练图像目录不存在，跳过真实图像测试")
        return False
    
    # 获取前几个图像文件
    image_files = list(train_images_path.glob("*.jpg"))[:3]
    if not image_files:
        print("❌ 未找到图像文件，跳过真实图像测试")
        return False
    
    try:
        from models.trained_model import load_trained_model
        
        model = load_trained_model()
        if model is None:
            print("❌ 无法加载模型进行真实图像测试")
            return False
        
        print(f"✅ 找到 {len(image_files)} 张测试图像")
        
        for i, image_file in enumerate(image_files):
            print(f"\n📸 测试图像 {i+1}: {image_file.name}")
            
            # 加载图像
            image = Image.open(image_file).convert('RGB')
            
            # 进行预测
            start_time = time.time()
            result = model.predict(image)
            prediction_time = time.time() - start_time
            
            print(f"  - 预测结果: {result['food_name']}")
            print(f"  - 置信度: {result['confidence']:.3f}")
            print(f"  - 预测时间: {prediction_time:.3f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 真实图像测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🧪 训练好的YOLO+ResNet融合模型测试")
    print("=" * 60)
    
    # 检查模型文件
    model_dir = Path("outputs/training/models")
    if not model_dir.exists():
        print("❌ 模型目录不存在，请先训练模型")
        return
    
    model_files = list(model_dir.glob("*.pth"))
    if not model_files:
        print("❌ 未找到模型文件，请先训练模型")
        return
    
    print(f"✅ 找到 {len(model_files)} 个模型文件:")
    for model_file in model_files:
        print(f"  - {model_file.name}")
    
    # 运行测试
    tests = [
        ("模型加载", test_model_loading),
        ("单张预测", test_prediction),
        ("批量预测", test_batch_prediction),
        ("真实图像", test_with_real_image)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} 测试通过")
        else:
            print(f"❌ {test_name} 测试失败")
    
    # 总结
    print(f"\n{'='*60}")
    print(f"📊 测试总结: {passed}/{total} 通过")
    if passed == total:
        print("🎉 所有测试通过！模型工作正常")
    else:
        print("⚠️ 部分测试失败，请检查模型和配置")
    print("=" * 60)

if __name__ == '__main__':
    main() 