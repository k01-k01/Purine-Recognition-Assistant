#!/usr/bin/env python3
"""
Food20_new数据集模型训练脚本
用于训练YOLO和ResNet50模型
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import yaml
from pathlib import Path
import logging
from ultralytics import YOLO
import json
import shutil
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet_classifier import ResNetClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Food20Dataset(Dataset):
    """Food20数据集加载器"""
    
    def __init__(self, data_dir, json_file, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        # 加载JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.images = data['images']
        self.annotations = data.get('annotations', [])
        
        # 创建图像ID到文件名的映射
        self.id_to_filename = {img['id']: img['file_name'] for img in self.images}
        
        # 创建类别映射
        self.categories = data.get('categories', [])
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        self.category_name_to_id = {cat['name']: cat['id'] for cat in self.categories}
        
        # 创建图像ID到类别的映射
        self.image_to_category = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            category_id = ann['category_id']
            if image_id not in self.image_to_category:
                self.image_to_category[image_id] = category_id
        
        logger.info(f"加载{self.mode}数据集: {len(self.images)}张图像, {len(self.category_id_to_name)}个类别")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        filename = img_info['file_name']
        
        # 加载图像
        img_path = os.path.join(self.data_dir, 'images', filename)
        image = Image.open(img_path).convert('RGB')
        
        # 获取类别
        category_id = self.image_to_category.get(img_id, 0)
        category_name = self.category_id_to_name.get(category_id, 'unknown')
        
        if self.transform:
            image = self.transform(image)
        
        return image, category_id, category_name

class ModelTrainer:
    """Food20模型训练器"""
    
    def __init__(self, config_path: str = "config/food20_training_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def _load_config(self) -> dict:
        """加载训练配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                "resnet": {
                    "epochs": 50,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "num_classes": 20,  # Food20有20个类别
                    "model_save_path": "models/resnet_food20_classifier.pth"
                },
                "yolo": {
                    "epochs": 100,
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "model_save_path": "models/yolo_food20_detector.pt"
                },
                "data": {
                    "dataset_path": "datasets/Food20_new",
                    "train_path": "datasets/Food20_new/train",
                    "val_path": "datasets/Food20_new/val",
                    "test_path": "datasets/Food20_new/test"
                }
            }
    
    def prepare_yolo_dataset(self):
        """准备YOLO格式的数据集"""
        logger.info("准备YOLO格式的数据集...")
        
        # 自动查找真实数据集路径
        config_path = self.config["data"]["dataset_path"]
        dataset_path = config_path if os.path.isabs(config_path) else os.path.abspath(config_path)
        if not os.path.exists(dataset_path):
            # 递归查找父目录
            found = False
            search_root = os.path.abspath('.')
            for root, dirs, files in os.walk(search_root):
                if os.path.basename(dataset_path) in dirs:
                    candidate = os.path.join(root, os.path.basename(dataset_path))
                    if os.path.exists(candidate):
                        dataset_path = candidate
                        found = True
                        logger.info(f"自动定位到数据集目录: {dataset_path}")
                        break
            if not found:
                logger.warning(f"数据集目录未找到: {dataset_path}")
        # ...existing code...
        
        # 创建YOLO数据集目录结构
        yolo_dataset_path = "datasets/food20_yolo"
        os.makedirs(yolo_dataset_path, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(yolo_dataset_path, split)
            os.makedirs(split_path, exist_ok=True)
            os.makedirs(os.path.join(split_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(split_path, 'labels'), exist_ok=True)
        
        # 处理每个分割
        for split in ['train', 'val', 'test']:
            source_path = os.path.join(dataset_path, split)
            target_path = os.path.join(yolo_dataset_path, split)
            
            if not os.path.exists(source_path):
                logger.warning(f"分割 {split} 不存在: {source_path}")
                continue
            
            # 加载JSON文件
            json_file = os.path.join(source_path, f"{split}.json")
            if not os.path.exists(json_file):
                logger.warning(f"JSON文件不存在: {json_file}")
                continue
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 创建类别映射
            categories = data.get('categories', [])
            category_id_to_name = {cat['id']: cat['name'] for cat in categories}
            
            # 创建图像ID到标注的映射
            image_annotations = {}
            for ann in data.get('annotations', []):
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)
            
            # 处理每张图像
        # 处理每个分割，收集所有类别
        all_categories = []
        for split in ['train', 'val', 'test']:
            source_path = os.path.join(dataset_path, split)
            target_path = os.path.join(yolo_dataset_path, split)
            if not os.path.exists(source_path):
                logger.warning(f"分割 {split} 不存在: {source_path}")
                continue
            json_file = os.path.join(source_path, f"{split}.json")
            if not os.path.exists(json_file):
                logger.warning(f"JSON文件不存在: {json_file}")
                continue
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            categories = data.get('categories', [])
            if not all_categories:
                all_categories = categories
            # ...existing code for category_id_to_name, image_annotations, 处理每张图像...
            category_id_to_name = {cat['id']: cat['name'] for cat in categories}
            image_annotations = {}
            for ann in data.get('annotations', []):
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)
            for img_info in data['images']:
                img_id = img_info['id']
                filename = img_info['file_name']
                width = img_info['width']
                height = img_info['height']
                src_img_path = os.path.join(source_path, 'images', filename)
                dst_img_path = os.path.join(target_path, 'images', filename)
                if os.path.exists(src_img_path):
                    shutil.copy2(src_img_path, dst_img_path)
                    label_filename = filename.replace('.jpg', '.txt')
                    label_path = os.path.join(target_path, 'labels', label_filename)
                    annotations = image_annotations.get(img_id, [])
                    with open(label_path, 'w') as f:
                        for ann in annotations:
                            bbox = ann['bbox']
                            x, y, w, h = bbox
                            x_center = (x + w / 2) / width
                            y_center = (y + h / 2) / height
                            w_rel = w / width
                            h_rel = h / height
                            category_id = ann['category_id'] - 1
                            f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {w_rel:.6f} {h_rel:.6f}\n")
        # 创建YOLO数据集配置文件
        if not all_categories:
            raise RuntimeError("未能收集到任何类别信息，YOLO数据集准备失败！")
        yaml_content = f"""
# Food20 YOLO数据集配置
path: {os.path.abspath(yolo_dataset_path)}
train: train/images
val: val/images
test: test/images

# 类别数
nc: {len(all_categories)}

# 类别名称
names:
"""
        for cat in all_categories:
            yaml_content += f"  {cat['id']-1}: {cat['name']}\n"
        yaml_path = os.path.join(yolo_dataset_path, "dataset.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        logger.info(f"YOLO数据集准备完成: {yolo_dataset_path}")
        return yaml_path
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        train_dir = os.path.join(dataset_path, 'train')
        train_json = os.path.join(train_dir, 'train.json')
        val_dir = os.path.join(dataset_path, 'val')
        val_json = os.path.join(val_dir, 'val.json')
        train_dataset = Food20Dataset(
            train_dir,
            train_json,
            transform=transform,
            mode='train'
        )
        val_dataset = Food20Dataset(
            val_dir,
            val_json,
            transform=val_transform,
            mode='val'
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4)
        
        # 创建模型
        model = ResNetClassifier(num_classes=config["num_classes"])
        model.model.train()
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.model.parameters(), lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 训练循环
        best_val_acc = 0.0
        
        for epoch in range(3):
            # 训练阶段
            model.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target, _) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1}/3, Batch {batch_idx}/{len(train_loader)}, "
                              f"Loss: {loss.item():.4f}")
            
            # 验证阶段
            model.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target, _ in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            logger.info(f"Epoch {epoch+1}/3: "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, "
                       f"Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss/len(val_loader):.4f}, "
                       f"Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.model.state_dict(), config["model_save_path"])
                logger.info(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
            
            scheduler.step()
        
        logger.info(f"ResNet训练完成！最佳验证准确率: {best_val_acc:.2f}%")
    
    def train_yolo(self):
        """训练YOLO检测器"""
        logger.info("开始训练YOLO检测器...")
        
        config = self.config["yolo"]
        
        # 准备YOLO数据集
        yaml_path = self.prepare_yolo_dataset()
        
        # 创建YOLO模型
        model = YOLO("yolov8n.pt")  # 使用YOLOv8n作为基础模型
        
        # 训练模型
        try:
            results = model.train(
                data=yaml_path,
                epochs=config["epochs"],
                batch=config["batch_size"],
                imgsz=640,
                device=self.device,
                project="models",
                name="yolo_food20_detector",
                save=True,
                save_period=10
            )
            
            # 保存最终模型
            model.save(config["model_save_path"])
            logger.info(f"YOLO训练完成！模型保存到: {config['model_save_path']}")
            
        except Exception as e:
            logger.error(f"YOLO训练失败: {e}")
    
    def evaluate_models(self):
        """评估训练好的模型"""
        logger.info("开始评估模型...")
        
        # 加载测试数据集
        dataset_path = self.config["data"]["dataset_path"]
        test_dataset = Food20Dataset(
            os.path.join(dataset_path, 'test'),
            os.path.join(dataset_path, 'test', 'test.json'),
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            mode='test'
        )
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # 评估ResNet模型
        resnet_path = self.config["resnet"]["model_save_path"]
        if os.path.exists(resnet_path):
            logger.info("评估ResNet模型...")
            model = ResNetClassifier(num_classes=self.config["resnet"]["num_classes"])
            model.model.load_state_dict(torch.load(resnet_path, map_location=self.device))
            model.model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target, _ in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model.model(data)
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            accuracy = 100. * correct / total
            logger.info(f"ResNet测试准确率: {accuracy:.2f}%")
        
        # 评估YOLO模型
        yolo_path = self.config["yolo"]["model_save_path"]
        if os.path.exists(yolo_path):
            logger.info("评估YOLO模型...")
            yolo_model = YOLO(yolo_path)
            
            # 在测试集上运行验证
            yaml_path = "datasets/food20_yolo/dataset.yaml"
            if os.path.exists(yaml_path):
                results = yolo_model.val(data=yaml_path)
                logger.info(f"YOLO mAP: {results.box.map}")

def main():
    parser = argparse.ArgumentParser(description="训练Food20食物识别模型")
    parser.add_argument("--model", choices=["resnet", "yolo", "both"], default="both",
                       help="选择要训练的模型")
    parser.add_argument("--config", default="config/food20_training_config.yaml",
                       help="训练配置文件路径")
    parser.add_argument("--prepare-only", action="store_true",
                       help="只准备数据集，不训练")
    parser.add_argument("--evaluate", action="store_true",
                       help="评估训练好的模型")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(args.config)
    
    if args.prepare_only:
        logger.info("准备YOLO数据集...")
        trainer.prepare_yolo_dataset()
        logger.info("数据集准备完成")
        return
    
    if args.evaluate:
        trainer.evaluate_models()
        return
    
    # 训练模型
    if args.model in ["resnet", "both"]:
        trainer.train_resnet()
    
    if args.model in ["yolo", "both"]:
        trainer.train_yolo()
    
    logger.info("模型训练完成！")

if __name__ == "__main__":
    main() 