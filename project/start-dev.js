const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 启动嘌呤识别助手开发环境...\n');

// 启动前端服务
console.log('📱 启动前端服务 (Next.js)...');
const frontend = spawn('npm', ['run', 'dev'], {
  cwd: process.cwd(),
  stdio: 'inherit',
  shell: true
});

// 等待2秒后启动后端服务
setTimeout(() => {
  console.log('\n🔧 启动后端服务 (FastAPI)...');
  const backend = spawn('python', ['start_server.py'], {
    cwd: path.join(process.cwd(), 'backend'),
    stdio: 'inherit',
    shell: true
  });

  backend.on('error', (error) => {
    console.error('❌ 后端服务启动失败:', error);
  });

  backend.on('close', (code) => {
    console.log(`\n🔧 后端服务已停止 (退出码: ${code})`);
  });
}, 2000);

frontend.on('error', (error) => {
  console.error('❌ 前端服务启动失败:', error);
});

frontend.on('close', (code) => {
  console.log(`\n📱 前端服务已停止 (退出码: ${code})`);
});

// 优雅关闭
process.on('SIGINT', () => {
  console.log('\n🛑 正在关闭所有服务...');
  frontend.kill('SIGINT');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 正在关闭所有服务...');
  frontend.kill('SIGTERM');
  process.exit(0);
});

console.log('\n✅ 开发环境启动完成！');
console.log('📱 前端服务: http://localhost:3000');
console.log('🔧 后端服务: http://localhost:3003');
console.log('📖 API文档: http://localhost:3003/docs');
console.log('\n按 Ctrl+C 停止所有服务\n'); 