#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

// 创建测试图片
function createTestImage() {
  const { createCanvas } = require('canvas');
  
  // 创建一个简单的测试图片
  const canvas = createCanvas(400, 300);
  const ctx = canvas.getContext('2d');
  
  // 绘制背景
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, 400, 300);
  
  // 绘制一个简单的食物图标
  ctx.fillStyle = 'orange';
  ctx.beginPath();
  ctx.arc(200, 150, 80, 0, 2 * Math.PI);
  ctx.fill();
  
  ctx.strokeStyle = 'red';
  ctx.lineWidth = 3;
  ctx.stroke();
  
  // 添加文字
  ctx.fillStyle = 'black';
  ctx.font = '24px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Food', 200, 160);
  
  // 保存为base64
  return canvas.toDataURL('image/jpeg').split(',')[1];
}

// 测试API
async function testAPI() {
  const baseURL = 'http://localhost:3003';
  
  console.log('🧪 开始测试食物识别API...\n');
  
  try {
    // 测试健康检查
    console.log('1. 测试健康检查...');
    const healthResponse = await fetch(`${baseURL}/health`);
    const healthData = await healthResponse.json();
    console.log('✅ 健康检查通过:', healthData);
    
    // 测试API信息
    console.log('\n2. 测试API信息...');
    const infoResponse = await fetch(`${baseURL}/api/info`);
    const infoData = await infoResponse.json();
    console.log('✅ API信息获取成功:', infoData.name);
    
    // 测试模型状态
    console.log('\n3. 测试模型状态...');
    const statusResponse = await fetch(`${baseURL}/api/ai-status`);
    const statusData = await statusResponse.json();
    console.log('✅ 模型状态:', statusData.data?.status || 'unknown');
    
    // 测试食物识别
    console.log('\n4. 测试食物识别...');
    
    // 创建测试图片
    let testImageData;
    try {
      testImageData = createTestImage();
    } catch (error) {
      console.log('⚠️  无法创建测试图片，使用默认base64数据');
      // 使用一个简单的base64图片数据
      testImageData = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
    }
    
    const recognitionResponse = await fetch(`${baseURL}/api/recognize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: testImageData
      })
    });
    
    const recognitionData = await recognitionResponse.json();
    
    if (recognitionData.success) {
      console.log('✅ 食物识别成功:');
      console.log(`   食物名称: ${recognitionData.data.foodName}`);
      console.log(`   嘌呤等级: ${recognitionData.data.purineLevel}`);
      console.log(`   置信度: ${recognitionData.data.confidence}`);
      console.log(`   建议: ${recognitionData.data.advice}`);
    } else {
      console.log('❌ 食物识别失败:', recognitionData.error);
    }
    
    // 测试食物数据库
    console.log('\n5. 测试食物数据库...');
    const foodsResponse = await fetch(`${baseURL}/api/foods`);
    const foodsData = await foodsResponse.json();
    console.log(`✅ 食物数据库获取成功，共${foodsData.data?.length || 0}种食物`);
    
    console.log('\n🎉 所有测试完成！');
    
  } catch (error) {
    console.error('❌ 测试失败:', error.message);
    console.log('\n💡 请确保服务已启动: npm run dev');
  }
}

// 运行测试
if (require.main === module) {
  testAPI();
}

module.exports = { testAPI }; 