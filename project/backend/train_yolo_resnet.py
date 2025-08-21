#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO+ResNet50 融合模型训练脚本
用于Food20_new数据集的训练
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import time
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class FoodDataset(Dataset):
    """食物数据集类"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform=None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据集根目录
            split: 数据集分割 ('train', 'val', 'test')
            transform: 图像变换
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # 加载标注文件
        annotation_file = self.data_dir / split / f'{split}.json'
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 获取类别映射
        self.classes = list(set([ann['category'] for ann in self.annotations['annotations']]))
        self.classes.sort()  # 确保类别顺序一致
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # 构建图像路径和标签
        self.images = []
        self.labels = []
        
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            category = ann['category']
            
            # 查找对应的图像信息
            image_info = next((img for img in self.annotations['images'] if img['id'] == image_id), None)
            if image_info:
                image_path = self.data_dir / split / 'images' / image_info['file_name']
                if image_path.exists():
                    self.images.append(str(image_path))
                    self.labels.append(self.class_to_idx[category])
        
        logger.info(f"📊 {split}数据集加载完成: {len(self.images)} 张图像, {len(self.classes)} 个类别")
        logger.info(f"📋 类别列表: {self.classes}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class YOLOResNetFusion(nn.Module):
    """YOLO+ResNet融合模型"""
    
    def __init__(self, num_classes: int, yolo_model_path: str = None):
        """
        初始化融合模型
        
        Args:
            num_classes: 分类类别数
            yolo_model_path: YOLO模型路径
        """
        super(YOLOResNetFusion, self).__init__()
        
        # ResNet50特征提取器
        self.resnet = models.resnet50(pretrained=True)
        # 移除最后的分类层
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # YOLO检测器
        self.yolo = YOLO('yolov8n.pt') if yolo_model_path is None else YOLO(yolo_model_path)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 检测特征维度
        self.detection_features = 1024  # YOLO检测特征维度
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(2048 + self.detection_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        logger.info(f"🔧 融合模型初始化完成: {num_classes} 个类别")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, 3, H, W]
        
        Returns:
            分类结果
        """
        batch_size = x.size(0)
        
        # ResNet特征提取
        resnet_features = self.resnet_features(x)
        resnet_features = resnet_features.view(batch_size, -1)  # [batch_size, 2048]
        
        # YOLO检测特征 (简化版本，实际应用中需要更复杂的处理)
        # 这里我们使用ResNet特征的变体作为检测特征
        detection_features = resnet_features[:, :self.detection_features]
        
        # 特征融合
        combined_features = torch.cat([resnet_features, detection_features], dim=1)
        
        # 分类
        output = self.classifier(combined_features)
        
        return output

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🚀 使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        # 数据变换
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 初始化模型和训练组件
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
    def load_data(self):
        """加载数据"""
        logger.info("📂 开始加载数据...")
        
        # 训练数据
        train_dataset = FoodDataset(
            data_dir=self.config['data_dir'],
            split='train',
            transform=self.train_transform
        )
        
        # 验证数据
        val_dataset = FoodDataset(
            data_dir=self.config['data_dir'],
            split='val',
            transform=self.val_transform
        )
        
        # 数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.num_classes = len(train_dataset.classes)
        self.classes = train_dataset.classes
        
        logger.info(f"✅ 数据加载完成: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本")
        
        return train_dataset, val_dataset
    
    def setup_model(self):
        """设置模型"""
        logger.info("🔧 开始设置模型...")
        
        # 创建模型
        self.model = YOLOResNetFusion(
            num_classes=self.num_classes,
            yolo_model_path=self.config.get('yolo_model_path')
        ).to(self.device)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['lr_step_size'],
            gamma=self.config['lr_gamma']
        )
        
        logger.info("✅ 模型设置完成")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def train(self):
        """完整训练流程"""
        logger.info("🚀 开始训练...")
        
        # 加载数据
        train_dataset, val_dataset = self.load_data()
        
        # 设置模型
        self.setup_model()
        
        # 训练历史
        train_history = {'loss': [], 'accuracy': []}
        val_history = {'loss': [], 'accuracy': []}
        best_val_acc = 0.0
        
        # 训练循环
        for epoch in range(self.config['epochs']):
            logger.info(f"📈 Epoch {epoch+1}/{self.config['epochs']}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            train_history['loss'].append(train_metrics['loss'])
            train_history['accuracy'].append(train_metrics['accuracy'])
            
            # 验证
            val_metrics = self.validate_epoch()
            val_history['loss'].append(val_metrics['loss'])
            val_history['accuracy'].append(val_metrics['accuracy'])
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录日志
            logger.info(f"📊 Epoch {epoch+1} - Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                self.save_model('best_model.pth')
                logger.info(f"💾 保存最佳模型，验证准确率: {best_val_acc:.2f}%")
            
            # 定期保存检查点
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
        
        # 保存最终模型
        self.save_model('final_model.pth')
        
        # 绘制训练曲线
        self.plot_training_curves(train_history, val_history)
        
        # 生成分类报告
        self.generate_classification_report(val_metrics['predictions'], val_metrics['targets'])
        
        logger.info("✅ 训练完成!")
    
    def save_model(self, filename: str):
        """保存模型"""
        model_path = self.output_dir / 'models' / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'classes': self.classes,
            'num_classes': self.num_classes
        }, model_path)
        logger.info(f"💾 模型已保存到: {model_path}")
    
    def plot_training_curves(self, train_history: Dict, val_history: Dict):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(train_history['loss'], label='Train Loss')
        ax1.plot(val_history['loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(train_history['accuracy'], label='Train Accuracy')
        ax2.plot(val_history['accuracy'], label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 训练曲线已保存")
    
    def generate_classification_report(self, predictions: List, targets: List):
        """生成分类报告"""
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        
        # 分类报告
        report = classification_report(targets, predictions, target_names=self.classes, output_dict=True)
        
        # 保存报告
        report_path = self.output_dir / 'classification_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 混淆矩阵
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📋 分类报告已生成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO+ResNet50 融合模型训练')
    parser.add_argument('--data_dir', type=str, default='datasets/Food20_new',
                       help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='outputs/training',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--lr_step_size', type=int, default=20,
                       help='学习率调度步长')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                       help='学习率调度衰减因子')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='模型保存间隔')
    
    args = parser.parse_args()
    
    # 训练配置
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'lr_step_size': args.lr_step_size,
        'lr_gamma': args.lr_gamma,
        'num_workers': args.num_workers,
        'save_interval': args.save_interval,
        'yolo_model_path': None  # 使用默认YOLOv8n模型
    }
    
    # 创建训练器并开始训练
    trainer = ModelTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main() 