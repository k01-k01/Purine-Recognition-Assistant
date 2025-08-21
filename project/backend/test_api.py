#!/usr/bin/env python3
"""
API测试脚本
用于测试食物识别API的功能
"""

import requests
import base64
import json
import time
from pathlib import Path
import argparse

class APITester:
    """API测试器"""
    
    def __init__(self, base_url: str = "http://localhost:3003"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self):
        """测试健康检查"""
        print("=== 测试健康检查 ===")
        try:
            response = self.session.get(f"{self.base_url}/health")
            print(f"状态码: {response.status_code}")
            print(f"响应: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False
    
    def test_api_info(self):
        """测试API信息"""
        print("\n=== 测试API信息 ===")
        try:
            response = self.session.get(f"{self.base_url}/api/info")
            print(f"状态码: {response.status_code}")
            data = response.json()
            print(f"API名称: {data.get('name')}")
            print(f"版本: {data.get('version')}")
            print(f"模型状态: {data.get('model', {}).get('status')}")
            return response.status_code == 200
        except Exception as e:
            print(f"API信息测试失败: {e}")
            return False
    
    def test_model_status(self):
        """测试模型状态"""
        print("\n=== 测试模型状态 ===")
        try:
            response = self.session.get(f"{self.base_url}/api/model-status")
            print(f"状态码: {response.status_code}")
            data = response.json()
            print(f"成功: {data.get('success')}")
            if data.get('data'):
                print(f"模型类型: {data['data'].get('model_type')}")
                print(f"可用性: {data['data'].get('available')}")
            return response.status_code == 200
        except Exception as e:
            print(f"模型状态测试失败: {e}")
            return False
    
    def test_foods_database(self):
        """测试食物数据库"""
        print("\n=== 测试食物数据库 ===")
        try:
            response = self.session.get(f"{self.base_url}/api/foods")
            print(f"状态码: {response.status_code}")
            data = response.json()
            print(f"成功: {data.get('success')}")
            if data.get('data'):
                print(f"食物数量: {len(data['data'])}")
                for food in data['data'][:3]:  # 显示前3个食物
                    print(f"  - {food.get('name')}: {food.get('purineLevel')}")
            return response.status_code == 200
        except Exception as e:
            print(f"食物数据库测试失败: {e}")
            return False
    
    def test_food_info(self, food_name: str = "apple"):
        """测试特定食物信息"""
        print(f"\n=== 测试食物信息: {food_name} ===")
        try:
            response = self.session.get(f"{self.base_url}/api/foods/{food_name}")
            print(f"状态码: {response.status_code}")
            data = response.json()
            print(f"成功: {data.get('success')}")
            if data.get('data'):
                food = data['data']
                print(f"食物名称: {food.get('name')}")
                print(f"嘌呤等级: {food.get('purineLevel')}")
                print(f"嘌呤含量: {food.get('purineContent')}")
                print(f"适合痛风患者: {food.get('suitableForGout')}")
            return response.status_code == 200
        except Exception as e:
            print(f"食物信息测试失败: {e}")
            return False
    
    def test_upload_image(self, image_path: str):
        """测试图片上传"""
        print(f"\n=== 测试图片上传: {image_path} ===")
        try:
            if not Path(image_path).exists():
                print(f"图片文件不存在: {image_path}")
                return False
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/api/upload", files=files)
            
            print(f"状态码: {response.status_code}")
            data = response.json()
            print(f"成功: {data.get('success')}")
            if data.get('data'):
                print(f"文件名: {data['data'].get('filename')}")
                print(f"文件大小: {data['data'].get('size')} bytes")
                return data['data'].get('dataUrl')
            return None
        except Exception as e:
            print(f"图片上传测试失败: {e}")
            return None
    
    def test_recognition(self, image_data: str):
        """测试食物识别"""
        print("\n=== 测试食物识别 ===")
        try:
            payload = {"image": image_data}
            response = self.session.post(
                f"{self.base_url}/api/recognize",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"状态码: {response.status_code}")
            data = response.json()
            print(f"成功: {data.get('success')}")
            
            if data.get('data'):
                result = data['data']
                print(f"识别结果:")
                print(f"  食物名称: {result.get('foodName')}")
                print(f"  嘌呤等级: {result.get('purineLevel')}")
                print(f"  嘌呤含量: {result.get('purineContent')}")
                print(f"  适合痛风患者: {result.get('suitableForGout')}")
                print(f"  置信度: {result.get('confidence')}")
                print(f"  建议: {result.get('advice')}")
                
                nutrition = result.get('nutritionEstimate', {})
                print(f"  营养成分:")
                print(f"    热量: {nutrition.get('calories')}")
                print(f"    蛋白质: {nutrition.get('protein')}")
                print(f"    脂肪: {nutrition.get('fat')}")
                print(f"    碳水化合物: {nutrition.get('carbohydrates')}")
                print(f"    膳食纤维: {nutrition.get('fiber')}")
            
            return response.status_code == 200
        except Exception as e:
            print(f"食物识别测试失败: {e}")
            return False
    
    def test_with_sample_image(self, image_path: str):
        """使用示例图片进行完整测试"""
        print(f"\n=== 使用示例图片测试: {image_path} ===")
        
        # 上传图片
        image_data = self.test_upload_image(image_path)
        if not image_data:
            print("图片上传失败，跳过识别测试")
            return False
        
        # 提取base64数据
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # 识别食物
        return self.test_recognition(image_data)
    
    def run_all_tests(self, image_path: str = None):
        """运行所有测试"""
        print("开始API测试...")
        print(f"测试地址: {self.base_url}")
        
        tests = [
            ("健康检查", self.test_health),
            ("API信息", self.test_api_info),
            ("模型状态", self.test_model_status),
            ("食物数据库", self.test_foods_database),
            ("食物信息", lambda: self.test_food_info("apple")),
        ]
        
        if image_path:
            tests.append(("图片上传和识别", lambda: self.test_with_sample_image(image_path)))
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*50}")
            start_time = time.time()
            success = test_func()
            end_time = time.time()
            duration = end_time - start_time
            
            results.append({
                "test": test_name,
                "success": success,
                "duration": duration
            })
            
            status = "✅ 通过" if success else "❌ 失败"
            print(f"{test_name}: {status} ({duration:.2f}s)")
        
        # 输出测试总结
        print(f"\n{'='*50}")
        print("测试总结:")
        passed = sum(1 for r in results if r["success"])
        total = len(results)
        print(f"总测试数: {total}")
        print(f"通过数: {passed}")
        print(f"失败数: {total - passed}")
        print(f"成功率: {passed/total*100:.1f}%")
        
        return passed == total

def create_sample_image():
    """创建示例图片（如果不存在）"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # 创建一个简单的食物图片
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制一个简单的食物图标
        draw.ellipse([100, 100, 300, 200], fill='orange', outline='red', width=3)
        draw.text((150, 150), "Food", fill='black')
        
        # 保存图片
        sample_path = "sample_food.jpg"
        img.save(sample_path)
        print(f"创建示例图片: {sample_path}")
        return sample_path
    except ImportError:
        print("PIL未安装，无法创建示例图片")
        return None

def main():
    parser = argparse.ArgumentParser(description="测试食物识别API")
    parser.add_argument("--url", default="http://localhost:3003", help="API基础URL")
    parser.add_argument("--image", help="测试图片路径")
    parser.add_argument("--create-sample", action="store_true", help="创建示例图片")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    # 创建示例图片
    if args.create_sample:
        image_path = create_sample_image()
    else:
        image_path = args.image
    
    # 运行测试
    success = tester.run_all_tests(image_path)
    
    if success:
        print("\n🎉 所有测试通过！")
        exit(0)
    else:
        print("\n⚠️  部分测试失败，请检查服务状态")
        exit(1)

if __name__ == "__main__":
    main() 