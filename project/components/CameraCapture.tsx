'use client';

import { useState, useRef, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Camera, RotateCw, X, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

interface CameraCaptureProps {
  onCapture: (imageSrc: string) => void;
  onClose: () => void;
}

export default function CameraCapture({ onCapture, onClose }: CameraCaptureProps) {
  const webcamRef = useRef<Webcam>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');
  const [cameraError, setCameraError] = useState<string>('');

  const capture = useCallback(() => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setCapturedImage(imageSrc);
      }
    }
  }, [webcamRef]);

  const retake = () => {
    setCapturedImage(null);
  };

  const confirmCapture = () => {
    if (capturedImage) {
      onCapture(capturedImage);
    }
  };

  const switchCamera = () => {
    setFacingMode(prev => prev === 'user' ? 'environment' : 'user');
  };

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: facingMode,
  };

  const handleUserMediaError = (error: string | DOMException) => {
    console.error('相机访问失败:', error);
    setCameraError('无法访问相机，请检查相机权限或尝试刷新页面');
  };

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-black text-white">
        <Button
          variant="ghost"
          size="sm"
          onClick={onClose}
          className="text-white hover:bg-white/10"
        >
          <X className="w-5 h-5" />
        </Button>
        <h3 className="text-lg font-medium">拍照识别</h3>
        <Button
          variant="ghost"
          size="sm"
          onClick={switchCamera}
          className="text-white hover:bg-white/10"
        >
          <RotateCw className="w-5 h-5" />
        </Button>
      </div>

      {/* Camera View */}
      <div className="flex-1 relative">
        {cameraError ? (
          <div className="flex items-center justify-center h-full bg-gray-900">
            <div className="text-center text-white p-6">
              <Camera className="w-16 h-16 mx-auto mb-4 text-gray-400" />
              <p className="text-lg font-medium mb-2">相机访问失败</p>
              <p className="text-sm text-gray-300 mb-4">{cameraError}</p>
              <Button
                onClick={onClose}
                variant="outline"
                className="text-white border-white hover:bg-white hover:text-black"
              >
                返回
              </Button>
            </div>
          </div>
        ) : !capturedImage ? (
          <div className="relative w-full h-full">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              className="w-full h-full object-cover"
              onUserMediaError={handleUserMediaError}
            />
            
            {/* Camera Overlay */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-64 h-64 border-2 border-white border-dashed rounded-lg flex items-center justify-center">
                <Camera className="w-12 h-12 text-white/50" />
              </div>
            </div>

            {/* Camera Controls */}
            <div className="absolute bottom-8 left-0 right-0 flex justify-center">
              <Button
                onClick={capture}
                size="lg"
                className="w-16 h-16 rounded-full bg-white hover:bg-gray-100"
              >
                <div className="w-12 h-12 rounded-full bg-emerald-600"></div>
              </Button>
            </div>
          </div>
        ) : (
          <div className="relative w-full h-full">
            <img
              src={capturedImage}
              alt="Captured"
              className="w-full h-full object-cover"
            />
            
            {/* Image Controls */}
            <div className="absolute bottom-8 left-0 right-0 flex justify-center space-x-4">
              <Button
                onClick={retake}
                variant="outline"
                size="lg"
                className="bg-white/90 hover:bg-white"
              >
                重拍
              </Button>
              <Button
                onClick={confirmCapture}
                size="lg"
                className="bg-emerald-600 hover:bg-emerald-700"
              >
                <Check className="w-5 h-5 mr-2" />
                确认
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Instructions */}
      {!capturedImage && (
        <div className="p-4 bg-black/80 text-white text-center">
          <p className="text-sm">
            将食物放在取景框内，点击下方按钮拍照
          </p>
        </div>
      )}
    </div>
  );
} 