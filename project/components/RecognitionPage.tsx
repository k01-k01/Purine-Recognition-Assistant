'use client';

import { useState, useRef } from 'react';
import { Camera, Upload, RotateCcw, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import CameraCapture from './CameraCapture';
import { recognizeFood } from '@/lib/recognition-api';

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

export default function RecognitionPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<RecognitionResult | null>(null);
  const [error, setError] = useState<string>('');
  const [showCamera, setShowCamera] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // 验证文件类型
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setError('请选择JPG、PNG或WEBP格式的图片');
      return;
    }

    // 验证文件大小
    if (file.size > 8 * 1024 * 1024) {
      setError('图片大小不能超过8MB');
      return;
    }

    setError('');
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImage(e.target?.result as string);
      setResult(null);
    };
    reader.readAsDataURL(file);
  };

  const handleRecognizeFood = async () => {
    if (!selectedImage) return;

    setIsLoading(true);
    setError('');

    try {
      const result = await recognizeFood(selectedImage);
      setResult(result);
    } catch (err) {
      setError('识别失败，请重试');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCameraCapture = (imageSrc: string) => {
    setSelectedImage(imageSrc);
    setShowCamera(false);
    setResult(null);
  };

  const getPurineLevelColor = (level: string) => {
    switch (level) {
      case 'high': return 'text-red-600 bg-red-50 border-red-200';
      case 'medium': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getPurineLevelIcon = (level: string) => {
    switch (level) {
      case 'high': return <XCircle className="w-5 h-5" />;
      case 'medium': return <AlertCircle className="w-5 h-5" />;
      case 'low': return <CheckCircle className="w-5 h-5" />;
      default: return null;
    }
  };

  const reset = () => {
    setSelectedImage(null);
    setResult(null);
    setError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="p-4 space-y-6">
      {/* 使用说明 */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-blue-500 mt-0.5" />
            <div>
              <h3 className="font-medium text-gray-900 mb-1">使用说明</h3>
              <p className="text-sm text-gray-600">
                拍摄或上传食物图片，AI将自动识别食物种类并提供嘌呤含量信息。
                支持JPG、PNG、WEBP格式，图片大小不超过8MB。
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 错误提示 */}
      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertCircle className="h-4 w-4 text-red-600" />
          <AlertDescription className="text-red-600">{error}</AlertDescription>
        </Alert>
      )}

      {/* 图片上传区域 */}
      {!selectedImage ? (
        <div className="space-y-4">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center bg-white">
            <Camera className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 mb-4">选择图片来识别食物嘌呤含量</p>
            <div className="space-y-3">
              <Button 
                onClick={() => setShowCamera(true)}
                className="w-full bg-emerald-600 hover:bg-emerald-700"
              >
                <Camera className="w-4 h-4 mr-2" />
                拍照识别
              </Button>
              <Button 
                onClick={() => fileInputRef.current?.click()}
                variant="outline"
                className="w-full"
              >
                <Upload className="w-4 h-4 mr-2" />
                从相册选择
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={handleImageSelect}
                className="hidden"
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* 图片预览 */}
          <Card>
            <CardContent className="p-4">
              <div className="relative">
                <img
                  src={selectedImage}
                  alt="Selected food"
                  className="w-full h-64 object-cover rounded-lg"
                />
                <Button
                  onClick={reset}
                  variant="outline"
                  size="sm"
                  className="absolute top-2 right-2 bg-white"
                >
                  <RotateCcw className="w-4 h-4" />
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* 识别按钮 */}
          {!result && (
            <Button
              onClick={handleRecognizeFood}
              disabled={isLoading}
              className="w-full bg-emerald-600 hover:bg-emerald-700"
            >
              {isLoading ? '识别中...' : '开始识别'}
            </Button>
          )}

          {/* 识别结果 */}
          {result && (
            <Card>
              <CardContent className="p-4 space-y-4">
                <div className="text-center">
                  <h3 className="text-lg font-bold text-gray-900 mb-2">识别结果</h3>
                  <div className="text-2xl font-bold text-emerald-600 mb-1">
                    {result.foodName}
                  </div>
                  <div className="text-sm text-gray-500">
                    识别置信度: {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="space-y-3">
                  {/* 嘌呤等级 */}
                  <div className={`flex items-center justify-between p-3 rounded-lg border ${getPurineLevelColor(result.purineLevel)}`}>
                    <div className="flex items-center space-x-2">
                      {getPurineLevelIcon(result.purineLevel)}
                      <span className="font-medium">嘌呤含量</span>
                    </div>
                    <div className="text-right">
                      <div className="font-bold">
                        {result.purineLevel === 'high' ? '高' : 
                         result.purineLevel === 'medium' ? '中' : '低'}
                      </div>
                      <div className="text-sm">{result.purineContent}</div>
                    </div>
                  </div>

                  {/* 适合性 */}
                  <div className={`flex items-center justify-between p-3 rounded-lg border ${
                    result.suitableForGout 
                      ? 'text-green-600 bg-green-50 border-green-200' 
                      : 'text-red-600 bg-red-50 border-red-200'
                  }`}>
                    <div className="flex items-center space-x-2">
                      {result.suitableForGout ? (
                        <CheckCircle className="w-5 h-5" />
                      ) : (
                        <XCircle className="w-5 h-5" />
                      )}
                      <span className="font-medium">适合痛风患者</span>
                    </div>
                    <div className="text-right">
                      <div className="font-bold">
                        {result.suitableForGout ? '适合' : '不适合'}
                      </div>
                    </div>
                  </div>

                  {/* 营养成分 */}
                  <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                    <div className="flex items-start space-x-2">
                      <div className="w-5 h-5 bg-gray-400 rounded-full mt-0.5" />
                      <div className="flex-1">
                        <div className="font-medium text-gray-900 mb-2">营养成分 (每100g)</div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-gray-600">热量:</span>
                            <span className="font-medium">{result.nutritionEstimate.calories}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">蛋白质:</span>
                            <span className="font-medium">{result.nutritionEstimate.protein}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">脂肪:</span>
                            <span className="font-medium">{result.nutritionEstimate.fat}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">碳水化合物:</span>
                            <span className="font-medium">{result.nutritionEstimate.carbohydrates}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">膳食纤维:</span>
                            <span className="font-medium">{result.nutritionEstimate.fiber}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 饮食建议 */}
                  <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                    <div className="flex items-start space-x-2">
                      <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                      <div>
                        <div className="font-medium text-blue-900 mb-1">饮食建议</div>
                        <p className="text-sm text-blue-700">{result.advice}</p>
                      </div>
                    </div>
                  </div>
                </div>

                <Button
                  onClick={reset}
                  variant="outline"
                  className="w-full"
                >
                  重新识别
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Camera Capture Modal */}
      {showCamera && (
        <CameraCapture
          onCapture={handleCameraCapture}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  );
}