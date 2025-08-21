const fetch = require('node-fetch');

async function testIntegration() {
  console.log('🧪 测试前后端集成...\n');

  const backendUrl = 'http://localhost:3003';
  const frontendUrl = 'http://localhost:3000';

  // 测试后端健康检查
  console.log('1. 测试后端健康检查...');
  try {
    const healthResponse = await fetch(`${backendUrl}/health`);
    if (healthResponse.ok) {
      const healthData = await healthResponse.json();
      console.log('✅ 后端健康检查通过:', healthData);
    } else {
      console.log('❌ 后端健康检查失败');
      return;
    }
  } catch (error) {
    console.log('❌ 后端服务不可用:', error.message);
    return;
  }

  // 测试API信息
  console.log('\n2. 测试API信息...');
  try {
    const infoResponse = await fetch(`${backendUrl}/api/info`);
    if (infoResponse.ok) {
      const infoData = await infoResponse.json();
      console.log('✅ API信息获取成功:', infoData.name);
      console.log('   模型状态:', infoData.model.status);
    } else {
      console.log('❌ API信息获取失败');
    }
  } catch (error) {
    console.log('❌ API信息获取失败:', error.message);
  }

  // 测试食物数据库
  console.log('\n3. 测试食物数据库...');
  try {
    const foodsResponse = await fetch(`${backendUrl}/api/foods`);
    if (foodsResponse.ok) {
      const foodsData = await foodsResponse.json();
      console.log('✅ 食物数据库获取成功');
      console.log('   食物数量:', foodsData.data?.length || 0);
    } else {
      console.log('❌ 食物数据库获取失败');
    }
  } catch (error) {
    console.log('❌ 食物数据库获取失败:', error.message);
  }

  // 测试食物识别（模拟）
  console.log('\n4. 测试食物识别...');
  try {
    // 创建一个简单的base64图片数据
    const mockImage = Buffer.from('fake_image_data').toString('base64');
    
    const recognitionResponse = await fetch(`${backendUrl}/api/recognize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image: mockImage
      })
    });

    if (recognitionResponse.ok) {
      const recognitionData = await recognitionResponse.json();
      console.log('✅ 食物识别测试成功');
      console.log('   识别结果:', recognitionData.data?.foodName);
      console.log('   嘌呤等级:', recognitionData.data?.purineLevel);
    } else {
      console.log('❌ 食物识别测试失败');
    }
  } catch (error) {
    console.log('❌ 食物识别测试失败:', error.message);
  }

  // 测试前端连接
  console.log('\n5. 测试前端连接...');
  try {
    const frontendResponse = await fetch(frontendUrl);
    if (frontendResponse.ok) {
      console.log('✅ 前端服务可访问');
    } else {
      console.log('❌ 前端服务不可访问');
    }
  } catch (error) {
    console.log('❌ 前端服务不可访问:', error.message);
  }

  console.log('\n🎉 集成测试完成！');
  console.log('\n📝 使用说明:');
  console.log('1. 前端地址: http://localhost:3000');
  console.log('2. 后端地址: http://localhost:3003');
  console.log('3. API文档: http://localhost:3003/docs');
  console.log('4. 健康检查: http://localhost:3003/health');
}

// 运行测试
testIntegration().catch(console.error); 