const http = require('http');

// 测试健康检查端点
function testHealth() {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: 'localhost',
      port: 3001,
      path: '/health',
      method: 'GET',
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        console.log('健康检查响应:', data);
        resolve(data);
      });
    });

    req.on('error', (err) => {
      console.error('请求失败:', err.message);
      reject(err);
    });

    req.end();
  });
}

// 测试服务器信息端点
function testInfo() {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: 'localhost',
      port: 3001,
      path: '/api/info',
      method: 'GET',
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        console.log('服务器信息响应:', data);
        resolve(data);
      });
    });

    req.on('error', (err) => {
      console.error('请求失败:', err.message);
      reject(err);
    });

    req.end();
  });
}

// 运行测试
async function runTests() {
  console.log('开始测试API...\n');
  
  try {
    await testHealth();
    console.log('\n---\n');
    await testInfo();
    console.log('\nAPI测试完成！');
  } catch (error) {
    console.error('测试失败:', error);
  }
}

runTests(); 