from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import os
from dotenv import load_dotenv
from services.food_recognition_service import FoodRecognitionService
from models.schemas import RecognitionRequest, RecognitionResponse, FoodInfo
from utils.food_database import FoodDatabase
import logging
import time

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="食物识别与尿酸分析API",
    description="基于YOLO+ResNet融合模型的食物识别和尿酸含量分析服务",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/public", StaticFiles(directory="public"), name="public")

# 初始化服务
food_recognition_service = None
food_database = FoodDatabase()

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    global food_recognition_service
    try:
        logger.info("🚀 应用启动，开始初始化服务...")
        food_recognition_service = FoodRecognitionService()
        logger.info("✅ 食物识别服务初始化成功")
    except Exception as e:
        logger.error(f"❌ 食物识别服务初始化失败: {e}")
        food_recognition_service = None

@app.get("/health")
async def health_check():
    """健康检查"""
    logger.info("🏥 健康检查请求")
    return {
        "status": "ok",
        "timestamp": "2024-01-01T00:00:00Z",
        "service": "food-recognition-api"
    }

@app.get("/api/info")
async def get_api_info():
    """获取API信息"""
    logger.info("📋 API信息请求")
    model_status = "available" if food_recognition_service else "unavailable"
    
    return {
        "name": "食物识别与尿酸分析API",
        "version": "2.0.0",
        "description": "基于YOLO+ResNet融合模型的食物识别和尿酸含量分析服务",
        "model": {
            "type": "YOLO+ResNet融合模型",
            "status": model_status,
            "available": food_recognition_service is not None
        },
        "endpoints": [
            "GET /health - 健康检查",
            "GET /api/info - API信息",
            "GET /api/model-status - 模型状态",
            "POST /api/recognize - 食物识别",
            "POST /api/upload - 图片上传",
            "GET /api/foods - 食物数据库",
            "GET /api/foods/{food_name} - 获取特定食物信息"
        ]
    }

@app.get("/api/model-status")
async def get_model_status():
    """获取模型状态"""
    logger.info("📊 模型状态请求")
    if not food_recognition_service:
        logger.warning("⚠️ 模型服务未初始化")
        return {
            "success": False,
            "error": "模型服务未初始化",
            "status": "unavailable"
        }
    
    try:
        status = await food_recognition_service.get_status()
        logger.info("✅ 模型状态获取成功")
        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        logger.error(f"❌ 模型状态检查失败: {str(e)}")
        return {
            "success": False,
            "error": f"模型状态检查失败: {str(e)}",
            "status": "error"
        }

@app.post("/api/recognize", response_model=RecognitionResponse)
async def recognize_food(request: RecognitionRequest):
    """食物识别API"""
    start_time = time.time()
    logger.info("🍽️ 收到食物识别请求")
    
    try:
        if not food_recognition_service:
            logger.error("❌ 模型服务不可用")
            raise HTTPException(status_code=503, detail="模型服务不可用")
        
        if not request.image:
            logger.error("❌ 图片数据为空")
            raise HTTPException(status_code=400, detail="图片数据不能为空")
        
        logger.info(f"📸 开始处理图片，数据长度: {len(request.image)} 字符")
        
        # 调用识别服务
        result = await food_recognition_service.recognize_food(request.image)
        
        total_time = time.time() - start_time
        logger.info(f"✅ 食物识别完成: {result.foodName} (耗时: {total_time:.3f}秒)")
        
        return RecognitionResponse(
            success=True,
            data=result
        )
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"❌ 食物识别失败 (耗时: {total_time:.3f}秒): {e}")
        return RecognitionResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """图片上传API"""
    logger.info(f"📤 收到图片上传请求: {file.filename}")
    
    try:
        # 检查文件类型
        if not file.content_type.startswith('image/'):
            logger.error(f"❌ 不支持的文件类型: {file.content_type}")
            raise HTTPException(status_code=400, detail="只能上传图片文件")
        
        # 检查文件大小 (限制为8MB)
        if file.size > 8 * 1024 * 1024:
            logger.error(f"❌ 文件过大: {file.size} bytes")
            raise HTTPException(status_code=400, detail="文件大小不能超过8MB")
        
        # 读取文件内容
        contents = await file.read()
        
        # 转换为base64
        import base64
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        logger.info(f"✅ 图片上传成功: {file.filename} ({len(contents)} bytes)")
        
        return {
            "success": True,
            "data": {
                "filename": file.filename,
                "mimetype": file.content_type,
                "size": len(contents),
                "dataUrl": f"data:{file.content_type};base64,{base64_image}"
            }
        }
    except Exception as e:
        logger.error(f"❌ 文件上传失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/foods")
async def get_foods():
    """获取食物数据库"""
    logger.info("📋 食物数据库请求")
    try:
        foods = food_database.get_all_foods()
        logger.info(f"✅ 食物数据库获取成功，共 {len(foods)} 种食物")
        return {
            "success": True,
            "data": foods
        }
    except Exception as e:
        logger.error(f"❌ 获取食物数据库失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/foods/{food_name}")
async def get_food_info(food_name: str):
    """获取特定食物信息"""
    logger.info(f"🔍 食物信息请求: {food_name}")
    try:
        food_info = food_database.get_food_info(food_name)
        if not food_info:
            logger.warning(f"⚠️ 未找到食物: {food_name}")
            raise HTTPException(status_code=404, detail=f"未找到食物: {food_name}")
        
        logger.info(f"✅ 食物信息获取成功: {food_name}")
        return {
            "success": True,
            "data": food_info
        }
    except Exception as e:
        logger.error(f"❌ 获取食物信息失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3003))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 启动食物识别API服务器...")
    print(f"📍 地址: http://{host}:{port}")
    print(f"📚 API文档: http://{host}:{port}/docs")
    print(f"🏥 健康检查: http://{host}:{port}/health")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # 关闭自动重载，避免频繁重启
        log_level="info"
    ) 