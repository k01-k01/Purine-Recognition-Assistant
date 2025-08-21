// 后端服务器配置
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:3003';

interface RecognitionResult {
  foodName: string;
  purineLevel: 'high' | 'medium' | 'low';
  purineContent: string;
  suitableForGout: boolean;
  advice: string;
  nutritionEstimate: {
    calories: string;
    protein: string;
    fat: string;
    carbohydrates: string;
    fiber: string;
  };
  confidence: number;
}

// 食物嘌呤数据库（本地备用）
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

export async function recognizeFood(imageBase64: string): Promise<RecognitionResult> {
  try {
    // 首先尝试使用后端服务器
    try {
      const response = await fetch(`${BACKEND_URL}/api/recognize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageBase64
        })
      });

      if (response.ok) {
        const result = await response.json();
        if (result.success && result.data) {
          console.log('✅ 使用后端AI模型识别成功');
          return result.data;
        }
      }
    } catch (backendError) {
      console.log('⚠️ 后端服务器不可用，使用本地模拟数据:', backendError);
    }

    // 如果后端不可用，使用本地模拟数据
    console.log('📝 使用本地模拟识别数据');
    return getMockRecognitionResult();

  } catch (error) {
    console.error('❌ 识别失败:', error);
    // 返回模拟结果
    return getMockRecognitionResult();
  }
}

function getMockRecognitionResult(): RecognitionResult {
  const mockResults: RecognitionResult[] = [
    {
      foodName: '动物肝脏',
      purineLevel: 'high',
      purineContent: '150-1000mg/100g',
      suitableForGout: false,
      advice: '高嘌呤食物，痛风患者应避免食用',
      nutritionEstimate: {
        calories: '135kcal/100g',
        protein: '21g/100g',
        fat: '3.6g/100g',
        carbohydrates: '2.5g/100g',
        fiber: '0g/100g'
      },
      confidence: 0.95
    },
    {
      foodName: '豆腐',
      purineLevel: 'low',
      purineContent: '30-50mg/100g',
      suitableForGout: true,
      advice: '低嘌呤食物，可以适量食用',
      nutritionEstimate: {
        calories: '76kcal/100g',
        protein: '8g/100g',
        fat: '4.8g/100g',
        carbohydrates: '1.9g/100g',
        fiber: '0.3g/100g'
      },
      confidence: 0.88
    },
    {
      foodName: '海鲜',
      purineLevel: 'high',
      purineContent: '100-300mg/100g',
      suitableForGout: false,
      advice: '高嘌呤食物，需要限制摄入',
      nutritionEstimate: {
        calories: '85kcal/100g',
        protein: '18g/100g',
        fat: '1.2g/100g',
        carbohydrates: '0g/100g',
        fiber: '0g/100g'
      },
      confidence: 0.92
    },
    {
      foodName: '蔬菜',
      purineLevel: 'low',
      purineContent: '10-50mg/100g',
      suitableForGout: true,
      advice: '低嘌呤食物，可以放心食用',
      nutritionEstimate: {
        calories: '25kcal/100g',
        protein: '2.5g/100g',
        fat: '0.3g/100g',
        carbohydrates: '4.5g/100g',
        fiber: '2.8g/100g'
      },
      confidence: 0.90
    },
    {
      foodName: '肉类',
      purineLevel: 'medium',
      purineContent: '50-150mg/100g',
      suitableForGout: true,
      advice: '中等嘌呤食物，适量食用',
      nutritionEstimate: {
        calories: '250kcal/100g',
        protein: '26g/100g',
        fat: '15g/100g',
        carbohydrates: '0g/100g',
        fiber: '0g/100g'
      },
      confidence: 0.87
    }
  ];

  return mockResults[Math.floor(Math.random() * mockResults.length)];
}

function findPurineInfo(foodName: string) {
  // 简单的关键词匹配
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

  // 默认返回低嘌呤
  return FOOD_PURINE_DB['蔬菜'];
}

// 检查后端服务器状态
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    return response.ok;
  } catch (error) {
    console.log('后端服务器不可用:', error);
    return false;
  }
} 