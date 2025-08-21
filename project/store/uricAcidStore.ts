'use client';

import { create } from 'zustand';
import { persist,createJSONStorage } from 'zustand/middleware';

interface UricAcidRecord {
  id: string;
  date: string;
  value: number;
  note: string;
}

interface UricAcidStore {
  records: UricAcidRecord[];
  addRecord: (record: Omit<UricAcidRecord, 'id'>) => void;
  deleteRecord: (id: string) => void;
  getStats: () => {
    average: number;
    max: number;
    min: number;
  };
}

export const useUricAcidStore = create<UricAcidStore>()(
  persist(
    (set, get) => ({
      records: [],
      
      addRecord: (record) => {
        const newRecord: UricAcidRecord = {
          ...record,
          id: Date.now().toString()
        };
        
        set((state) => ({
          records: [...state.records, newRecord].sort((a, b) => 
            new Date(b.date).getTime() - new Date(a.date).getTime()
          )
        }));
      },
      
      deleteRecord: (id) => {
        set((state) => ({
          records: state.records.filter(record => record.id !== id)
        }));
      },
      
      getStats: () => {
        const records = get().records;
        if (records.length === 0) {
          return { average: 0, max: 0, min: 0 };
        }
        
        // 取最近3次记录计算统计
        const recentRecords = records.slice(0, 3);
        const values = recentRecords.map(r => r.value);
        
        return {
          average: values.reduce((sum, val) => sum + val, 0) / values.length,
          max: Math.max(...values),
          min: Math.min(...values)
        };
      }
    }),
    {
      name: 'uric-acid-storage',
      storage: createJSONStorage(() => localStorage),
    }
  )
);