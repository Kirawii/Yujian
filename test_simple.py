#!/usr/bin/env python
"""
简单测试脚本 - 不加载模型，只测试API接口
"""

import requests
import sys
import os


def test_health(api_url: str = "http://localhost:8000"):
    """测试服务健康状态"""
    url = f"{api_url}/health"
    try:
        response = requests.get(url, timeout=5)
        print(f"健康检查结果: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False


def test_root(api_url: str = "http://localhost:8000"):
    """测试根路径"""
    url = f"{api_url}/"
    try:
        response = requests.get(url, timeout=5)
        print(f"根路径响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"根路径测试失败: {e}")
        return False


def test_video_translation(video_path: str, api_url: str = "http://localhost:8000"):
    """测试视频翻译"""
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在 {video_path}")
        return False

    url = f"{api_url}/translate/video"

    print(f"\n正在上传视频: {video_path}")
    print(f"文件大小: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")

    with open(video_path, "rb") as f:
        files = {"file": (os.path.basename(video_path), f, "video/mp4")}
        try:
            response = requests.post(url, files=files, timeout=300)
        except Exception as e:
            print(f"请求失败: {e}")
            return False

    if response.status_code == 200:
        result = response.json()
        print(f"\n翻译结果:")
        print(f"  成功: {result['success']}")
        print(f"  文本: {result['text']}")
        print(f"  消息: {result['message']}")
        return result['success']
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Uni-Sign API 简单测试")
    parser.add_argument("--mode", choices=["health", "root", "video", "all"], default="health",
                        help="测试模式")
    parser.add_argument("--file", type=str, help="要上传的视频文件路径")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                        help="API 服务地址")

    args = parser.parse_args()

    if args.mode == "health" or args.mode == "all":
        test_health(args.url)

    if args.mode == "root" or args.mode == "all":
        test_root(args.url)

    if args.mode == "video" or args.mode == "all":
        if not args.file:
            print("请使用 --file 指定视频文件路径")
            return
        test_video_translation(args.file, args.url)


if __name__ == "__main__":
    main()
