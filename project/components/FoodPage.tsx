'use client';

import { useState, useMemo } from 'react';
import { Search, Filter } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import foodData from '@/data/foods.json';

interface Food {
  id: string;
  name: string;
  category: string;
  purineLevel: 'high' | 'medium' | 'low';
  purineContent: string;
  description: string;
  image: string;
}

export default function FoodPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedLevel, setSelectedLevel] = useState<string>('all');

  const purineLevels = [
    { value: 'all', label: '全部', color: 'bg-gray-100 text-gray-700' },
    { value: 'low', label: '低嘌呤', color: 'bg-green-100 text-green-700' },
    { value: 'medium', label: '中嘌呤', color: 'bg-orange-100 text-orange-700' },
    { value: 'high', label: '高嘌呤', color: 'bg-red-100 text-red-700' }
  ];

  const filteredFoods = useMemo(() => {
    return foodData.filter((food) => {
      const matchesSearch = food.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           food.category.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesLevel = selectedLevel === 'all' || food.purineLevel === selectedLevel;
      return matchesSearch && matchesLevel;
    });
  }, [searchTerm, selectedLevel]);

  const getPurineLevelBadge = (level: string) => {
    switch (level) {
      case 'high':
        return <Badge className="bg-red-100 text-red-700">高嘌呤</Badge>;
      case 'medium':
        return <Badge className="bg-orange-100 text-orange-700">中嘌呤</Badge>;
      case 'low':
        return <Badge className="bg-green-100 text-green-700">低嘌呤</Badge>;
      default:
        return null;
    }
  };

  return (
    <div className="p-4 space-y-6">
      {/* 搜索栏 */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
        <Input
          placeholder="搜索食物或分类..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10"
        />
      </div>

      {/* 筛选栏 */}
      <div className="flex space-x-2 overflow-x-auto pb-2">
        {purineLevels.map((level) => (
          <Button
            key={level.value}
            variant={selectedLevel === level.value ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedLevel(level.value)}
            className={`whitespace-nowrap ${
              selectedLevel === level.value 
                ? 'bg-emerald-600 hover:bg-emerald-700' 
                : ''
            }`}
          >
            <Filter className="w-4 h-4 mr-1" />
            {level.label}
          </Button>
        ))}
      </div>

      {/* 统计信息 */}
      <div className="bg-emerald-50 p-4 rounded-lg border border-emerald-200">
        <div className="text-center">
          <div className="text-2xl font-bold text-emerald-600">
            {filteredFoods.length}
          </div>
          <div className="text-sm text-emerald-700">
            {selectedLevel === 'all' ? '种食物' : `种${purineLevels.find(l => l.value === selectedLevel)?.label}食物`}
          </div>
        </div>
      </div>

      {/* 食物列表 */}
      <div className="space-y-3">
        {filteredFoods.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-2">🔍</div>
            <p>未找到相关食物</p>
            <p className="text-sm">请尝试其他搜索词或筛选条件</p>
          </div>
        ) : (
          filteredFoods.map((food) => (
            <Card key={food.id} className="overflow-hidden">
              <CardContent className="p-0">
                <div className="flex">
                  {/* 食物图片 */}
                  <div className="w-20 h-20 bg-gray-100 flex-shrink-0">
                    <img
                      src={food.image}
                      alt={food.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                        target.parentElement!.innerHTML = `
                          <div class="w-full h-full bg-gray-200 flex items-center justify-center text-gray-400 text-2xl">
                            🍽️
                          </div>
                        `;
                      }}
                    />
                  </div>
                  
                  {/* 食物信息 */}
                  <div className="flex-1 p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h3 className="font-medium text-gray-900">{food.name}</h3>
                        <p className="text-sm text-gray-500">{food.category}</p>
                      </div>
                      {getPurineLevelBadge(food.purineLevel)}
                    </div>
                    
                    <div className="space-y-1">
                      <div className="text-sm text-gray-600">
                        <span className="font-medium">嘌呤含量：</span>
                        {food.purineContent}
                      </div>
                      <p className="text-sm text-gray-500">{food.description}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>

      {/* 加载更多提示 */}
      {filteredFoods.length > 0 && (
        <div className="text-center py-4 text-gray-500 text-sm">
          已显示全部结果
        </div>
      )}
    </div>
  );
}