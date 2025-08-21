#!/usr/bin/env python3
"""
Food20数据集测试脚本
用于验证数据集结构和数据加载
"""

import os
import json
import sys
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_structure(dataset_path):
    """测试数据集结构"""
    logger.info(f"测试数据集结构: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        logger.error(f"数据集路径不存在: {dataset_path}")
        return False
    
    # 检查各个分割
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            logger.error(f"分割 {split} 不存在: {split_path}")
            return False
        
        # 检查图像目录
        images_path = os.path.join(split_path, 'images')
        if not os.path.exists(images_path):
            logger.error(f"图像目录不存在: {images_path}")
            return False
        
        # 检查JSON文件
        json_file = os.path.join(split_path, f"{split}.json")
        if not os.path.exists(json_file):
            logger.error(f"JSON文件不存在: {json_file}")
            return False
        
        logger.info(f"✓ {split} 分割检查通过")
    
    return True

def test_json_files(dataset_path):
    """测试JSON文件格式"""
    logger.info("测试JSON文件格式...")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        json_file = os.path.join(dataset_path, split, f"{split}.json")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查必要字段
            required_fields = ['images']
            for field in required_fields:
                if field not in data:
                    logger.error(f"JSON文件缺少必要字段: {field}")
                    return False
            
            # 统计信息
            num_images = len(data['images'])
            num_annotations = len(data.get('annotations', []))
            num_categories = len(data.get('categories', []))
            
            logger.info(f"✓ {split}: {num_images}张图像, {num_annotations}个标注, {num_categories}个类别")
            
            # 检查图像文件是否存在
            images_path = os.path.join(dataset_path, split, 'images')
            missing_images = 0
            for img_info in data['images'][:10]:  # 只检查前10张
                img_path = os.path.join(images_path, img_info['file_name'])
                if not os.path.exists(img_path):
                    missing_images += 1
            
            if missing_images > 0:
                logger.warning(f"{split} 中有 {missing_images} 张图像文件缺失")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON文件格式错误: {e}")
            return False
        except Exception as e:
            logger.error(f"读取JSON文件失败: {e}")
            return False
    
    return True

def test_image_loading(dataset_path):
    """测试图像加载"""
    logger.info("测试图像加载...")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        json_file = os.path.join(dataset_path, split, f"{split}.json")
        images_path = os.path.join(dataset_path, split, 'images')
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 测试加载前5张图像
            for i, img_info in enumerate(data['images'][:5]):
                img_path = os.path.join(images_path, img_info['file_name'])
                
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path)
                        image.verify()  # 验证图像完整性
                        
                        # 重新打开图像获取信息
                        image = Image.open(img_path)
                        width, height = image.size
                        
                        # 检查尺寸是否匹配
                        if width != img_info.get('width', 0) or height != img_info.get('height', 0):
                            logger.warning(f"图像尺寸不匹配: {img_info['file_name']}")
                        
                        logger.info(f"✓ {split} 图像 {i+1}: {img_info['file_name']} ({width}x{height})")
                        
                    except Exception as e:
                        logger.error(f"加载图像失败 {img_info['file_name']}: {e}")
                        return False
                else:
                    logger.error(f"图像文件不存在: {img_path}")
                    return False
        
        except Exception as e:
            logger.error(f"测试 {split} 图像加载失败: {e}")
            return False
    
    return True

def test_annotations(dataset_path):
    """测试标注数据"""
    logger.info("测试标注数据...")
    
    splits = ['train', 'val', 'test']
    for split in splits:
        json_file = os.path.join(dataset_path, split, f"{split}.json")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            annotations = data.get('annotations', [])
            categories = data.get('categories', [])
            
            if annotations:
                # 检查标注格式
                for ann in annotations[:5]:  # 检查前5个标注
                    required_fields = ['image_id', 'category_id']
                    for field in required_fields:
                        if field not in ann:
                            logger.error(f"标注缺少必要字段: {field}")
                            return False
                    
                    # 检查边界框
                    if 'bbox' in ann:
                        bbox = ann['bbox']
                        if len(bbox) != 4:
                            logger.error(f"边界框格式错误: {bbox}")
                            return False
                
                logger.info(f"✓ {split}: 标注格式正确")
            else:
                logger.warning(f"{split}: 没有标注数据")
            
            if categories:
                logger.info(f"✓ {split}: 类别信息完整")
            else:
                logger.warning(f"{split}: 没有类别信息")
        
        except Exception as e:
            logger.error(f"测试 {split} 标注失败: {e}")
            return False
    
    return True

def main():
    """主函数"""
    dataset_path = "datasets/Food20_new"
    
    logger.info("开始测试Food20数据集...")
    
    # 测试数据集结构
    if not test_dataset_structure(dataset_path):
        logger.error("数据集结构测试失败")
        return False
    
    # 测试JSON文件
    if not test_json_files(dataset_path):
        logger.error("JSON文件测试失败")
        return False
    
    # 测试图像加载
    if not test_image_loading(dataset_path):
        logger.error("图像加载测试失败")
        return False
    
    # 测试标注数据
    if not test_annotations(dataset_path):
        logger.error("标注数据测试失败")
        return False
    
    logger.info("✓ 所有测试通过！数据集准备就绪。")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 