'use client';

import { useEffect, useRef } from 'react';

interface UricAcidRecord {
  id: string;
  date: string;
  value: number;
  note: string;
}

interface UricAcidChartProps {
  records: UricAcidRecord[];
}

export default function UricAcidChart({ records }: UricAcidChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<any>(null);

  useEffect(() => {
    let echarts: any;
    
    const loadECharts = async () => {
      if (typeof window !== 'undefined') {
        // 动态加载 ECharts
        const echartsModule = await import('echarts');
        echarts = echartsModule.default || echartsModule;
        
        if (chartRef.current) {
          // 销毁之前的图表实例
          if (chartInstance.current) {
            chartInstance.current.dispose();
          }
          
          // 创建新的图表实例
          chartInstance.current = echarts.init(chartRef.current);
          
          // 准备数据
          const sortedRecords = [...records].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
          const last30Records = sortedRecords.slice(-30); // 最近30条记录
          
          const dates = last30Records.map(record => {
            const date = new Date(record.date);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          });
          
          const values = last30Records.map(record => record.value);
          
          const option = {
            title: {
              show: false
            },
            tooltip: {
              trigger: 'axis',
              formatter: function(params: any) {
                const data = params[0];
                const originalRecord = last30Records[data.dataIndex];
                return `
                  <div style="padding: 8px;">
                    <div style="font-weight: bold; margin-bottom: 4px;">${new Date(originalRecord.date).toLocaleDateString('zh-CN')}</div>
                    <div style="color: ${data.value > 420 ? '#dc2626' : data.value > 360 ? '#ea580c' : '#16a34a'};">
                      尿酸值: ${data.value} μmol/L
                    </div>
                    ${originalRecord.note ? `<div style="font-size: 12px; color: #6b7280; margin-top: 4px;">${originalRecord.note}</div>` : ''}
                  </div>
                `;
              }
            },
            grid: {
              left: '3%',
              right: '4%',
              bottom: '10%',
              top: '15%',
              containLabel: true
            },
            xAxis: {
              type: 'category',
              data: dates,
              axisLine: {
                lineStyle: {
                  color: '#e5e7eb'
                }
              },
              axisLabel: {
                color: '#6b7280',
                fontSize: 11
              }
            },
            yAxis: {
              type: 'value',
              name: 'μmol/L',
              nameTextStyle: {
                color: '#6b7280',
                fontSize: 12
              },
              axisLine: {
                lineStyle: {
                  color: '#e5e7eb'
                }
              },
              axisLabel: {
                color: '#6b7280',
                fontSize: 11
              },
              splitLine: {
                lineStyle: {
                  color: '#f3f4f6',
                  type: 'dashed'
                }
              },
              // 添加参考线
              markLine: {
                data: [
                  {
                    yAxis: 420,
                    name: '高尿酸',
                    lineStyle: {
                      color: '#dc2626',
                      type: 'dashed'
                    },
                    label: {
                      formatter: '危险线 420',
                      position: 'end',
                      color: '#dc2626',
                      fontSize: 10
                    }
                  },
                  {
                    yAxis: 360,
                    name: '正常上限',
                    lineStyle: {
                      color: '#ea580c',
                      type: 'dashed'
                    },
                    label: {
                      formatter: '警戒线 360',
                      position: 'end',
                      color: '#ea580c',
                      fontSize: 10
                    }
                  }
                ]
              }
            },
            series: [
              {
                name: '尿酸值',
                type: 'line',
                data: values,
                smooth: true,
                symbol: 'circle',
                symbolSize: 6,
                lineStyle: {
                  color: '#059669',
                  width: 3
                },
                itemStyle: {
                  color: '#059669',
                  borderColor: '#ffffff',
                  borderWidth: 2
                },
                areaStyle: {
                  color: {
                    type: 'linear',
                    x: 0,
                    y: 0,
                    x2: 0,
                    y2: 1,
                    colorStops: [
                      {
                        offset: 0,
                        color: 'rgba(5, 150, 105, 0.3)'
                      },
                      {
                        offset: 1,
                        color: 'rgba(5, 150, 105, 0.05)'
                      }
                    ]
                  }
                }
              }
            ]
          };
          
          chartInstance.current.setOption(option);
          
          // 监听窗口大小变化
          const handleResize = () => {
            if (chartInstance.current) {
              chartInstance.current.resize();
            }
          };
          
          window.addEventListener('resize', handleResize);
          
          return () => {
            window.removeEventListener('resize', handleResize);
          };
        }
      }
    };

    loadECharts();

    return () => {
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
    };
  }, [records]);

  if (records.length === 0) {
    return null;
  }

  return (
    <div 
      ref={chartRef} 
      style={{ width: '100%', height: '250px' }}
      className="bg-white"
    />
  );
}