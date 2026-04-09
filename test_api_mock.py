#!/usr/bin/env python
"""
API 模拟测试 - 不加载真实模型，用于测试API接口结构
"""

import requests
import json

API_URL = "http://localhost:8000"


def test_api_structure():
    """测试API结构是否符合预期"""
    print("=" * 50)
    print("Uni-Sign API 结构测试")
    print("=" * 50)

    # 定义预期的API结构
    expected_structure = {
        "GET /": {
            "description": "服务信息",
            "response": {
                "service": "Uni-Sign 手语翻译服务",
                "version": "1.0.0",
                "endpoints": ["/translate/video", "/translate/image"]
            }
        },
        "GET /health": {
            "description": "健康检查",
            "response": {
                "status": "healthy",
                "model_loaded": True
            }
        },
        "POST /translate/video": {
            "description": "视频翻译",
            "request": {
                "file": "视频文件 (mp4/avi/mov等)"
            },
            "response": {
                "success": True,
                "text": "翻译后的文本",
                "message": "翻译成功"
            }
        },
        "POST /translate/image": {
            "description": "图片翻译",
            "request": {
                "file": "图片文件 (jpg/png等)"
            },
            "response": {
                "success": True,
                "text": "翻译后的文本",
                "message": "翻译成功"
            }
        }
    }

    print("\nAPI 结构定义:")
    for endpoint, info in expected_structure.items():
        print(f"\n{endpoint}")
        print(f"  描述: {info['description']}")
        if 'request' in info:
            print(f"  请求: {info['request']}")
        print(f"  响应: {info['response']}")

    return True


def test_with_curl_commands():
    """输出curl测试命令"""
    print("\n" + "=" * 50)
    print("使用 curl 测试 API")
    print("=" * 50)

    print("""
1. 健康检查:
   curl http://localhost:8000/health

2. 服务信息:
   curl http://localhost:8000/

3. 视频翻译:
   curl -X POST "http://localhost:8000/translate/video" \\
        -F "file=@./data/exam/example_1.mp4"

4. 图片翻译:
   curl -X POST "http://localhost:8000/translate/image" \\
        -F "file=@./demo/rtmlib-main/demo.jpg"
""")


def test_with_python():
    """输出Python测试代码"""
    print("\n" + "=" * 50)
    print("使用 Python requests 测试 API")
    print("=" * 50)

    code = '''
import requests

# 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 视频翻译
url = "http://localhost:8000/translate/video"
with open("./data/exam/example_1.mp4", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files, timeout=300)
    result = response.json()
    print(f"翻译结果: {result['text']}")

# 图片翻译
url = "http://localhost:8000/translate/image"
with open("./demo/rtmlib-main/demo.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files, timeout=60)
    result = response.json()
    print(f"翻译结果: {result['text']}")
'''
    print(code)


if __name__ == "__main__":
    test_api_structure()
    test_with_curl_commands()
    test_with_python()

    print("\n" + "=" * 50)
    print("测试说明")
    print("=" * 50)
    print("""
1. 首先启动服务:
   bash start_api.sh

2. 服务启动后会监听 http://0.0.0.0:8000

3. 使用上述 curl 或 Python 代码测试 API

4. 视频翻译可能需要几分钟时间，取决于:
   - 视频长度
   - 是否设置了 TARGET_FPS
   - 硬件性能 (CPU/GPU)

5. 如果服务无法启动，检查:
   - 模型权重文件是否存在: ./demo/pt/csl_daily_pose_only_slt.pth
   - 依赖是否安装完整
   - 显存/内存是否足够
""")
