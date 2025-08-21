#!/usr/bin/env python3
"""
FastAPI服务器启动脚本
"""

import uvicorn
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """启动FastAPI服务器"""
    # 设置环境变量
    os.environ.setdefault("PORT", "3003")
    os.environ.setdefault("HOST", "0.0.0.0")
    
    # 获取配置
    port = int(os.getenv("PORT", 3003))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"启动食物识别API服务器...")
    print(f"地址: http://{host}:{port}")
    print(f"API文档: http://{host}:{port}/docs")
    print(f"健康检查: http://{host}:{port}/health")
    
    # 启动服务器
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main() 