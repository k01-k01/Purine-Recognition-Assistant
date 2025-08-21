'use client';

import { useState } from 'react';
import { Camera, Database, User } from 'lucide-react';
import RecognitionPage from '@/components/RecognitionPage';
import FoodPage from '@/components/FoodPage';
import ProfilePage from '@/components/ProfilePage';

export default function Home() {
  const [activeTab, setActiveTab] = useState('recognition');

  const tabs = [
    { id: 'recognition', label: '识别', icon: Camera, component: RecognitionPage },
    { id: 'food', label: '食物', icon: Database, component: FoodPage },
    { id: 'profile', label: '我的', icon: User, component: ProfilePage },
  ];

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component || RecognitionPage;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-emerald-600 text-white">
        <div className="px-4 py-4">
          <h1 className="text-xl font-bold text-center">嘌呤识别助手</h1>
          <p className="text-emerald-100 text-sm text-center mt-1">科学饮食，健康生活</p>
        </div>
      </header>

      {/* Content Area */}
      <main className="pb-20">
        <ActiveComponent />
      </main>

      {/* Bottom Navigation */}
      <nav className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200">
        <div className="flex justify-around">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex flex-col items-center py-3 px-4 transition-colors ${
                  isActive 
                    ? 'text-emerald-600' 
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                <Icon className={`w-6 h-6 mb-1 ${isActive ? 'text-emerald-600' : ''}`} />
                <span className={`text-xs ${isActive ? 'font-medium' : ''}`}>
                  {tab.label}
                </span>
              </button>
            );
          })}
        </div>
      </nav>
    </div>
  );
}