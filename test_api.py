"""
Uni-Sign FastAPI 服务测试脚本
"""

import requests
import sys
import os


def test_video_translation(video_path: str, api_url: str = "http://localhost:8000"):
    """
    测试视频手语翻译

    Args:
        video_path: 视频文件路径
        api_url: API 服务地址
    """
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在 {video_path}")
        return

    url = f"{api_url}/translate/video"

    print(f"正在上传视频: {video_path}")
    with open(video_path, "rb") as f:
        files = {"file": (os.path.basename(video_path), f, "video/mp4")}
        response = requests.post(url, files=files, timeout=300)

    if response.status_code == 200:
        result = response.json()
        print(f"\n翻译结果:")
        print(f"  成功: {result['success']}")
        print(f"  文本: {result['text']}")
        print(f"  消息: {result['message']}")
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)


def test_image_translation(image_path: str, api_url: str = "http://localhost:8000"):
    """
    测试图片手语翻译

    Args:
        image_path: 图片文件路径
        api_url: API 服务地址
    """
    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 {image_path}")
        return

    url = f"{api_url}/translate/image"

    print(f"正在上传图片: {image_path}")
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(url, files=files, timeout=60)

    if response.status_code == 200:
        result = response.json()
        print(f"\n翻译结果:")
        print(f"  成功: {result['success']}")
        print(f"  文本: {result['text']}")
        print(f"  消息: {result['message']}")
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)


def test_health(api_url: str = "http://localhost:8000"):
    """测试服务健康状态"""
    url = f"{api_url}/health"
    try:
        response = requests.get(url, timeout=5)
        print(f"健康检查: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Uni-Sign API 测试脚本")
    parser.add_argument("--mode", choices=["video", "image", "health"], default="health",
                        help="测试模式")
    parser.add_argument("--file", type=str, help="要上传的文件路径")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                        help="API 服务地址")

    args = parser.parse_args()

    if args.mode == "health":
        test_health(args.url)
    elif args.mode == "video":
        if not args.file:
            print("请使用 --file 指定视频文件路径")
            return
        test_video_translation(args.file, args.url)
    elif args.mode == "image":
        if not args.file:
            print("请使用 --file 指定图片文件路径")
            return
        test_image_translation(args.file, args.url)


if __name__ == "__main__":
    main()
