#!/usr/bin/env python
"""
测试 FastAPI 路由结构 - 不加载模型
"""

from fastapi.testclient import TestClient
import sys

# 创建一个测试用的简化API
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="Uni-Sign 手语翻译服务",
    description="支持图片/视频转文字的手语翻译 API",
    version="1.0.0"
)


class TranslateResponse(BaseModel):
    success: bool
    text: str
    message: str


@app.get("/")
async def root():
    return {
        "service": "Uni-Sign 手语翻译服务",
        "version": "1.0.0",
        "endpoints": [
            "/translate/video - 上传视频进行手语翻译",
            "/translate/image - 上传图片进行手语翻译"
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": False  # 测试时模型未加载
    }


@app.post("/translate/video", response_model=TranslateResponse)
async def translate_video(file: UploadFile = File(...)):
    """视频翻译接口 - 测试版"""
    return TranslateResponse(
        success=True,
        text="这是一个测试响应 - 视频翻译功能",
        message=f"收到文件: {file.filename}"
    )


@app.post("/translate/image", response_model=TranslateResponse)
async def translate_image(file: UploadFile = File(...)):
    """图片翻译接口 - 测试版"""
    return TranslateResponse(
        success=True,
        text="这是一个测试响应 - 图片翻译功能",
        message=f"收到文件: {file.filename}"
    )


# 创建测试客户端
client = TestClient(app)


def test_root():
    """测试根路径"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Uni-Sign 手语翻译服务"
    assert data["version"] == "1.0.0"
    print("✓ 根路径测试通过")
    return data


def test_health():
    """测试健康检查"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("✓ 健康检查测试通过")
    return data


def test_video_upload():
    """测试视频上传接口"""
    # 创建一个假的视频文件
    fake_video = b"fake video content"
    response = client.post(
        "/translate/video",
        files={"file": ("test.mp4", fake_video, "video/mp4")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "text" in data
    print("✓ 视频上传接口测试通过")
    return data


def test_image_upload():
    """测试图片上传接口"""
    # 创建一个假的图片文件
    fake_image = b"fake image content"
    response = client.post(
        "/translate/image",
        files={"file": ("test.jpg", fake_image, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "text" in data
    print("✓ 图片上传接口测试通过")
    return data


if __name__ == "__main__":
    print("=" * 50)
    print("FastAPI 路由结构测试")
    print("=" * 50)

    try:
        # 运行所有测试
        root_data = test_root()
        print(f"  服务信息: {root_data['service']}")
        print(f"  可用端点:")
        for endpoint in root_data['endpoints']:
            print(f"    - {endpoint}")
        print()

        health_data = test_health()
        print(f"  状态: {health_data['status']}")
        print(f"  模型加载: {health_data['model_loaded']}")
        print()

        video_data = test_video_upload()
        print(f"  响应: {video_data}")
        print()

        image_data = test_image_upload()
        print(f"  响应: {image_data}")
        print()

        print("=" * 50)
        print("所有测试通过!")
        print("=" * 50)
        print("""
API 路由结构验证完成。要测试真实功能，请:

1. 确保模型权重文件存在:
   ./demo/pt/csl_daily_pose_only_slt.pth

2. 安装所有依赖:
   pip install -r api_requirements.txt

3. 启动服务:
   bash start_api.sh

4. 使用 test_api.py 或 curl 进行真实测试
        """)

    except AssertionError as e:
        print(f"✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
