#!/usr/bin/env python3
"""
Food20模型训练快速启动脚本
"""

import os
import sys
import subprocess
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """检查环境"""
    logger.info("检查训练环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("需要Python 3.8或更高版本")
        return False
    
    logger.info(f"✓ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠ CUDA不可用，将使用CPU训练")
    except ImportError:
        logger.warning("⚠ PyTorch未安装")
    
    return True

def install_dependencies():
    """安装依赖"""
    logger.info("安装训练依赖...")
    
    try:
        # 安装PyTorch
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ], check=True)
        
        # 安装其他依赖
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "ultralytics", "pyyaml", "pillow", "numpy", "opencv-python"
        ], check=True)
        
        logger.info("✓ 依赖安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"依赖安装失败: {e}")
        return False

def test_dataset():
    """测试数据集"""
    logger.info("测试数据集...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_food20_dataset.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ 数据集测试通过")
            return True
        else:
            logger.error(f"数据集测试失败: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"数据集测试异常: {e}")
        return False

def train_models(model_type="both", config_path=None):
    """训练模型"""
    logger.info(f"开始训练模型: {model_type}")
    
    cmd = [sys.executable, "scripts/train_food20_models.py", "--model", model_type]
    if config_path:
        cmd.extend(["--config", config_path])
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("✓ 模型训练完成")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"模型训练失败: {e}")
        return False

def evaluate_models():
    """评估模型"""
    logger.info("评估训练好的模型...")
    
    try:
        subprocess.run([
            sys.executable, "scripts/train_food20_models.py", "--evaluate"
        ], check=True)
        
        logger.info("✓ 模型评估完成")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"模型评估失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Food20模型训练快速启动")
    parser.add_argument("--model", choices=["resnet", "yolo", "both"], default="both",
                       help="选择要训练的模型")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--skip-install", action="store_true", help="跳过依赖安装")
    parser.add_argument("--skip-test", action="store_true", help="跳过数据集测试")
    parser.add_argument("--skip-evaluate", action="store_true", help="跳过模型评估")
    
    args = parser.parse_args()
    
    logger.info("========================================")
    logger.info("Food20模型训练快速启动")
    logger.info("========================================")
    
    # 检查环境
    if not check_environment():
        return False
    
    # 安装依赖
    if not args.skip_install:
        if not install_dependencies():
            return False
    
    # 测试数据集
    if not args.skip_test:
        if not test_dataset():
            return False
    
    # 训练模型
    if not train_models(args.model, args.config):
        return False
    
    # 评估模型
    if not args.skip_evaluate:
        if not evaluate_models():
            return False
    
    logger.info("========================================")
    logger.info("训练流程完成！")
    logger.info("========================================")
    logger.info("模型文件保存在 models/ 目录下")
    logger.info("日志文件保存在 logs/ 目录下")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 