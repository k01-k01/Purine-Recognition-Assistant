const { AIService } = require('./dist/services/ai-service');

async function testAIService() {
  try {
    console.log('正在初始化AI服务...');
    const aiService = new AIService();
    
    console.log('正在检查AI服务状态...');
    const status = await aiService.getServiceStatus();
    console.log('AI服务状态:', status);
    
    if (!status.available) {
      console.error('AI服务不可用');
      return;
    }
    
    console.log('AI服务初始化成功！');
    console.log('模型:', status.model);
    console.log('状态:', status.status);
    
  } catch (error) {
    console.error('AI服务测试失败:', error.message);
    console.log('\n请检查以下配置:');
    console.log('1. 确保在 .env 文件中设置了 DASHSCOPE_API_KEY');
    console.log('2. 确保API Key有效且有足够的配额');
    console.log('3. 检查网络连接是否正常');
  }
}

// 运行测试
testAIService(); 