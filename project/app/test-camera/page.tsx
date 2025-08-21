'use client';

import { useState } from 'react';
import CameraCapture from '@/components/CameraCapture';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

export default function TestCameraPage() {
  const [showCamera, setShowCamera] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);

  const handleCapture = (imageSrc: string) => {
    setCapturedImage(imageSrc);
    setShowCamera(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-md mx-auto space-y-6">
        <h1 className="text-2xl font-bold text-center">相机功能测试</h1>
        
        <Card>
          <CardContent className="p-6">
            <Button
              onClick={() => setShowCamera(true)}
              className="w-full bg-emerald-600 hover:bg-emerald-700"
            >
              测试相机拍照
            </Button>
          </CardContent>
        </Card>

        {capturedImage && (
          <Card>
            <CardContent className="p-4">
              <h3 className="font-medium mb-3">拍摄的照片：</h3>
              <img
                src={capturedImage}
                alt="Captured"
                className="w-full rounded-lg"
              />
              <Button
                onClick={() => setCapturedImage(null)}
                variant="outline"
                className="w-full mt-3"
              >
                清除照片
              </Button>
            </CardContent>
          </Card>
        )}

        {showCamera && (
          <CameraCapture
            onCapture={handleCapture}
            onClose={() => setShowCamera(false)}
          />
        )}
      </div>
    </div>
  );
} 