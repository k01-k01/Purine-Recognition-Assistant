const fetch = require('node-fetch');

async function testServer() {
          const baseUrl = 'http://localhost:3003';
  
  try {
    console.log('正在测试服务器...');
    
    // 测试健康检查
    console.log('\n1. 测试健康检查...');
    const healthResponse = await fetch(`${baseUrl}/health`);
    const healthData = await healthResponse.json();
    console.log('健康检查结果:', healthData);
    
    // 测试服务器信息
    console.log('\n2. 测试服务器信息...');
    const infoResponse = await fetch(`${baseUrl}/api/info`);
    const infoData = await infoResponse.json();
    console.log('服务器信息:', infoData);
    
    // 测试AI服务状态
    console.log('\n3. 测试AI服务状态...');
    const aiStatusResponse = await fetch(`${baseUrl}/api/ai-status`);
    const aiStatusData = await aiStatusResponse.json();
    console.log('AI服务状态:', aiStatusData);
    
    console.log('\n✅ 服务器测试完成！');
    
  } catch (error) {
    console.error('❌ 服务器测试失败:', error.message);
    console.log('\n请确保:');
    console.log('1. 服务器正在运行 (npm run dev)');
    console.log('2. 端口3001没有被占用');
    console.log('3. 网络连接正常');
  }
}

// 运行测试
testServer(); 