#!/usr/bin/env python3
"""
显示Food20数据集的类别
"""
import json

def show_categories():
    # 加载训练数据
    with open('datasets/Food20_new/train/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("Food20数据集包含的20个类别:")
    print("=" * 40)
    
    for cat in data['categories']:
        print(f"{cat['id']:2d}: {cat['name']}")
    
    print("=" * 40)
    print(f"总计: {len(data['categories'])}个类别")

if __name__ == "__main__":
    show_categories() 