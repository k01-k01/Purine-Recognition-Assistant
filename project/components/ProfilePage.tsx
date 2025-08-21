'use client';

import { useState, useEffect } from 'react';
import { Plus, TrendingUp, Calendar, Trash2, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { useUricAcidStore } from '@/store/uricAcidStore';
import UricAcidChart from '@/components/UricAcidChart';

interface UricAcidRecord {
  id: string;
  date: string;
  value: number;
  note: string;
}

export default function ProfilePage() {
  const { records, addRecord, deleteRecord, getStats } = useUricAcidStore();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newRecord, setNewRecord] = useState({
    date: new Date().toISOString().split('T')[0],
    value: '',
    note: ''
  });

  const stats = getStats();

  const handleAddRecord = () => {
    if (!newRecord.value) return;

    addRecord({
      date: newRecord.date,
      value: parseFloat(newRecord.value),
      note: newRecord.note
    });

    setNewRecord({
      date: new Date().toISOString().split('T')[0],
      value: '',
      note: ''
    });
    setIsDialogOpen(false);
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('zh-CN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getValueColor = (value: number) => {
    if (value > 420) return 'text-red-600';
    if (value > 360) return 'text-orange-600';
    return 'text-green-600';
  };

  return (
    <div className="p-4 space-y-6">
      {/* 折线图 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-lg">
            <TrendingUp className="w-5 h-5 mr-2 text-emerald-600" />
            尿酸变化趋势
          </CardTitle>
        </CardHeader>
        <CardContent>
          {records.length > 0 ? (
            <UricAcidChart records={records} />
          ) : (
            <div className="text-center py-8 text-gray-500">
              <BarChart3 className="w-12 h-12 mx-auto mb-2 text-gray-300" />
              <p>暂无数据</p>
              <p className="text-sm">添加第一条尿酸记录开始监测</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 数据统计 */}
      {records.length > 0 && (
        <div className="grid grid-cols-3 gap-3">
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-lg font-bold text-blue-600">{stats.average.toFixed(1)}</div>
              <div className="text-xs text-gray-500">平均值</div>
              <div className="text-xs text-gray-400">μmol/L</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <div className={`text-lg font-bold ${getValueColor(stats.max)}`}>{stats.max}</div>
              <div className="text-xs text-gray-500">最高值</div>
              <div className="text-xs text-gray-400">μmol/L</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <div className={`text-lg font-bold ${getValueColor(stats.min)}`}>{stats.min}</div>
              <div className="text-xs text-gray-500">最低值</div>
              <div className="text-xs text-gray-400">μmol/L</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* 添加记录按钮 */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogTrigger asChild>
          <Button className="w-full bg-emerald-600 hover:bg-emerald-700">
            <Plus className="w-4 h-4 mr-2" />
            添加尿酸记录
          </Button>
        </DialogTrigger>
        <DialogContent className="w-[95%] max-w-md">
          <DialogHeader>
            <DialogTitle>添加尿酸记录</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label htmlFor="date">日期</Label>
              <Input
                id="date"
                type="date"
                value={newRecord.date}
                onChange={(e) => setNewRecord({...newRecord, date: e.target.value})}
              />
            </div>
            <div>
              <Label htmlFor="value">尿酸值 (μmol/L)</Label>
              <Input
                id="value"
                type="number"
                placeholder="请输入尿酸值"
                value={newRecord.value}
                onChange={(e) => setNewRecord({...newRecord, value: e.target.value})}
              />
              <p className="text-xs text-gray-500 mt-1">
                正常范围：男性 208-428 μmol/L，女性 155-357 μmol/L
              </p>
            </div>
            <div>
              <Label htmlFor="note">备注（可选）</Label>
              <Input
                id="note"
                placeholder="添加备注..."
                value={newRecord.note}
                onChange={(e) => setNewRecord({...newRecord, note: e.target.value})}
              />
            </div>
            <Button 
              onClick={handleAddRecord} 
              className="w-full bg-emerald-600 hover:bg-emerald-700"
              disabled={!newRecord.value}
            >
              保存记录
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* 记录列表 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center text-lg">
            <Calendar className="w-5 h-5 mr-2 text-emerald-600" />
            历史记录 ({records.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {records.length === 0 ? (
            <div className="text-center py-6 text-gray-500">
              <Calendar className="w-12 h-12 mx-auto mb-2 text-gray-300" />
              <p>暂无记录</p>
              <p className="text-sm">点击上方按钮添加第一条记录</p>
            </div>
          ) : (
            <div className="space-y-3">
              {records.map((record) => (
                <div key={record.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <div className="text-sm text-gray-500">
                        {formatDate(record.date)}
                      </div>
                      <div className={`font-bold ${getValueColor(record.value)}`}>
                        {record.value} μmol/L
                      </div>
                    </div>
                    {record.note && (
                      <p className="text-sm text-gray-600 mt-1">{record.note}</p>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteRecord(record.id)}
                    className="text-red-500 hover:text-red-700 hover:bg-red-50"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 健康提示 */}
      <Card className="bg-blue-50 border-blue-200">
        <CardContent className="p-4">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
              <TrendingUp className="w-4 h-4 text-blue-600" />
            </div>
            <div>
              <h3 className="font-medium text-blue-900 mb-1">健康提示</h3>
              <p className="text-sm text-blue-700">
                定期监测尿酸水平，配合合理饮食和适当运动，有助于痛风的预防和控制。
                建议每1-2周记录一次尿酸值。
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}