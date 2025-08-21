import Fastify, { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify';
import cors from '@fastify/cors';
import multipart from '@fastify/multipart';
import staticPlugin from '@fastify/static';
import path from 'path';
import { AIService, FoodRecognitionResult } from './services/ai-service';

// 类型定义
interface RecognitionRequest {
  image: string; // base64编码的图片
}

interface RecognitionResponse {
  success: boolean;
  data?: FoodRecognitionResult;
  error?: string;
}

// 食物嘌呤数据库
const FOOD_PURINE_DB: Record<string, { level: 'high' | 'medium' | 'low'; content: string; advice: string }> = {
  '动物肝脏': { level: 'high', content: '150-1000mg/100g', advice: '高嘌呤食物，痛风患者应避免食用' },
  '海鲜': { level: 'high', content: '100-300mg/100g', advice: '高嘌呤食物，需要限制摄入' },
  '肉类': { level: 'medium', content: '50-150mg/100g', advice: '中等嘌呤食物，适量食用' },
  '豆腐': { level: 'low', content: '30-50mg/100g', advice: '低嘌呤食物，可以适量食用' },
  '蔬菜': { level: 'low', content: '10-50mg/100g', advice: '低嘌呤食物，可以放心食用' },
  '水果': { level: 'low', content: '5-30mg/100g', advice: '低嘌呤食物，可以放心食用' },
  '豆类': { level: 'medium', content: '50-100mg/100g', advice: '中等嘌呤食物，适量食用' },
  '蘑菇': { level: 'medium', content: '50-150mg/100g', advice: '中等嘌呤食物，适量食用' },
  '啤酒': { level: 'high', content: '10-20mg/100ml', advice: '高嘌呤饮品，痛风患者应避免' },
  '茶': { level: 'low', content: '1-5mg/100ml', advice: '低嘌呤饮品，可以放心饮用' },
};

// 创建Fastify实例
const fastify = Fastify({
  logger: {
    level: 'info',
    transport: {
      target: 'pino-pretty',
      options: {
        colorize: true,
        translateTime: 'HH:MM:ss Z',
        ignore: 'pid,hostname',
      },
    },
  },
});

// 初始化AI服务
let aiService: AIService;
try {
  aiService = new AIService();
  fastify.log.info('AI服务初始化成功');
} catch (error) {
  fastify.log.error('AI服务初始化失败:', error);
  aiService = null as any;
}

// 注册插件
async function registerPlugins() {
  // CORS支持
  await fastify.register(cors, {
    origin: true, // 允许所有来源，生产环境应该设置具体域名
    credentials: true,
  });

  // 文件上传支持
  await fastify.register(multipart, {
    limits: {
      fileSize: 8 * 1024 * 1024, // 8MB
    },
  });

  // 静态文件服务
  await fastify.register(staticPlugin, {
    root: path.join(__dirname, '../public'),
    prefix: '/public/',
  });
}

// 工具函数
function findPurineInfo(foodName: string) {
  const keywords = {
    '肝脏': '动物肝脏',
    '肝': '动物肝脏',
    '海鲜': '海鲜',
    '鱼': '海鲜',
    '虾': '海鲜',
    '蟹': '海鲜',
    '肉': '肉类',
    '牛肉': '肉类',
    '猪肉': '肉类',
    '羊肉': '肉类',
    '豆腐': '豆腐',
    '豆': '豆类',
    '蘑菇': '蘑菇',
    '香菇': '蘑菇',
    '蔬菜': '蔬菜',
    '菜': '蔬菜',
    '水果': '水果',
    '苹果': '水果',
    '橙子': '水果',
    '啤酒': '啤酒',
    '茶': '茶',
  };

  for (const [keyword, category] of Object.entries(keywords)) {
    if (foodName.includes(keyword)) {
      return FOOD_PURINE_DB[category];
    }
  }

  return FOOD_PURINE_DB['蔬菜'];
}

// 路由定义
async function registerRoutes() {
  // 健康检查
  fastify.get('/health', async (request: FastifyRequest, reply: FastifyReply) => {
    return { status: 'ok', timestamp: new Date().toISOString() };
  });

  // 获取服务器信息
  fastify.get('/api/info', async (request: FastifyRequest, reply: FastifyReply) => {
    const aiStatus = aiService ? await aiService.getServiceStatus() : { status: 'unavailable', model: 'none', available: false };
    
    return {
      name: '嘌呤识别助手后端服务',
      version: '1.0.0',
      description: '基于Fastify的食物嘌呤识别API服务',
      aiService: aiStatus,
      endpoints: [
        'GET /health - 健康检查',
        'GET /api/info - 服务器信息',
        'GET /api/ai-status - AI服务状态',
        'POST /api/recognize - 食物识别',
        'POST /api/upload - 图片上传',
        'GET /api/foods - 食物列表',
      ],
    };
  });

  // AI服务状态检查
  fastify.get('/api/ai-status', async (request: FastifyRequest, reply: FastifyReply) => {
    if (!aiService) {
      return {
        success: false,
        error: 'AI服务未初始化',
        status: 'unavailable',
      };
    }

    try {
      const status = await aiService.getServiceStatus();
      return {
        success: true,
        data: status,
      };
    } catch (error) {
      return {
        success: false,
        error: 'AI服务状态检查失败',
        status: 'error',
      };
    }
  });

  // 食物识别API
  fastify.post<{ Body: RecognitionRequest }>(
    '/api/recognize',
    {
      schema: {
        body: {
          type: 'object',
          required: ['image'],
          properties: {
            image: { type: 'string' },
          },
        },
        response: {
          200: {
            type: 'object',
            properties: {
              success: { type: 'boolean' },
              data: {
                type: 'object',
                properties: {
                  foodName: { type: 'string' },
                  purineLevel: { type: 'string', enum: ['high', 'medium', 'low'] },
                  purineContent: { type: 'string' },
                  suitableForGout: { type: 'boolean' },
                  advice: { type: 'string' },
                  nutritionEstimate: {
                    type: 'object',
                    properties: {
                      calories: { type: 'string' },
                      protein: { type: 'string' },
                      fat: { type: 'string' },
                      carbohydrates: { type: 'string' },
                      fiber: { type: 'string' },
                    },
                  },
                  confidence: { type: 'number' },
                },
              },
              error: { type: 'string' },
            },
          },
        },
      },
    },
    async (request: FastifyRequest<{ Body: RecognitionRequest }>, reply: FastifyReply): Promise<RecognitionResponse> => {
      try {
        const { image } = request.body;

        if (!image) {
          return {
            success: false,
            error: '图片数据不能为空',
          };
        }

        // 检查AI服务是否可用
        if (!aiService) {
          return {
            success: false,
            error: 'AI服务不可用，请检查配置',
          };
        }

        // 调用AI服务进行识别
        const result = await aiService.recognizeFood(image);

        return {
          success: true,
          data: result,
        };
      } catch (error) {
        fastify.log.error('识别失败:', error);
        return {
          success: false,
          error: error instanceof Error ? error.message : '识别失败，请重试',
        };
      }
    }
  );

  // 图片上传API
  fastify.post('/api/upload', async (request: FastifyRequest, reply: FastifyReply) => {
    try {
      const data = await request.file();

      if (!data) {
        return reply.code(400).send({
          success: false,
          error: '没有上传文件',
        });
      }

      // 检查文件类型
      if (!data.mimetype.startsWith('image/')) {
        return reply.code(400).send({
          success: false,
          error: '只能上传图片文件',
        });
      }

      // 读取文件内容并转换为base64
      const buffer = await data.toBuffer();
      const base64 = buffer.toString('base64');
      const dataUrl = `data:${data.mimetype};base64,${base64}`;

      return {
        success: true,
        data: {
          filename: data.filename,
          mimetype: data.mimetype,
          size: buffer.length,
          dataUrl,
        },
      };
    } catch (error) {
      fastify.log.error('文件上传失败:', error);
      return reply.code(500).send({
        success: false,
        error: '文件上传失败',
      });
    }
  });

  // 获取食物列表
  fastify.get('/api/foods', async (request: FastifyRequest, reply: FastifyReply) => {
    return {
      success: true,
      data: Object.entries(FOOD_PURINE_DB).map(([name, info]) => ({
        name,
        ...info,
      })),
    };
  });
}

// 启动服务器
async function start() {
  try {
    await registerPlugins();
    await registerRoutes();

    const port = process.env.PORT ? parseInt(process.env.PORT) : 3003;
    const host = process.env.HOST || '0.0.0.0';

    await fastify.listen({ port, host });
    fastify.log.info(`服务器运行在 http://${host}:${port}`);
    fastify.log.info('可用的API端点:');
    fastify.log.info('  GET  /health - 健康检查');
    fastify.log.info('  GET  /api/info - 服务器信息');
    fastify.log.info('  GET  /api/ai-status - AI服务状态');
    fastify.log.info('  POST /api/recognize - 食物识别');
    fastify.log.info('  POST /api/upload - 图片上传');
    fastify.log.info('  GET  /api/foods - 食物列表');
  } catch (err) {
    fastify.log.error('启动服务器失败:', err);
    console.error('详细错误信息:', err);
    process.exit(1);
  }
}

// 优雅关闭
process.on('SIGINT', async () => {
  fastify.log.info('正在关闭服务器...');
  await fastify.close();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  fastify.log.info('正在关闭服务器...');
  await fastify.close();
  process.exit(0);
});

// 启动服务器
start(); 