# Uni-Sign FastAPI 部署说明

本项目提供了基于 FastAPI 的手语翻译服务，支持图片和视频转文字功能。

## 文件说明

- `api.py` - FastAPI 服务主文件
- `start_api.sh` - 服务启动脚本
- `test_api.py` - API 测试脚本
- `api_requirements.txt` - API 服务依赖

## 安装依赖

```bash
pip install -r api_requirements.txt
```

## 配置

编辑 `start_api.sh` 设置环境变量：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `CHECKPOINT_PATH` | 模型检查点路径 | `./pretrained_weight/best_model.pth` |
| `DEVICE` | 运行设备 (cuda/cpu) | `cuda` |
| `RGB_SUPPORT` | 是否启用 RGB 辅助分支 | `false` |
| `MAX_LENGTH` | 最大序列长度 | `256` |
| `DATASET` | 数据集类型 (CSL_Daily/CSL_News/How2Sign/OpenASL) | `CSL_Daily` |
| `TARGET_FPS` | 视频采样帧率，空值表示不采样使用所有帧 | `""` |

### 关于视频帧采样

**默认行为（TARGET_FPS=""）**：
- 读取视频的所有帧
- 如果帧数 > `MAX_LENGTH` (默认256)，则随机采样256帧
- 适合短视频（< 10秒@25FPS）

**FPS采样模式（如 TARGET_FPS="5.0"）**：
- 按目标帧率均匀采样
- 5分钟视频 @5FPS = 1500帧 → 再按 `MAX_LENGTH` 采样到256帧
- 减少长视频的处理时间和显存占用

**建议配置**：
| 视频长度 | 建议 TARGET_FPS |
|---------|----------------|
| < 10秒 | 不设置 |
| 10-30秒 | 5-10 |
| 30秒-2分钟 | 3-5 |
| > 2分钟 | 2-3 |

## 启动服务

```bash
bash start_api.sh
```

服务将在 `http://0.0.0.0:8000` 启动。

## API 接口

### 1. 健康检查

```bash
GET /health
```

响应：
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. 视频翻译

```bash
POST /translate/video
Content-Type: multipart/form-data

file: <视频文件>
```

响应：
```json
{
  "success": true,
  "text": "翻译后的文本",
  "message": "翻译成功"
}
```

### 3. 图片翻译

```bash
POST /translate/image
Content-Type: multipart/form-data

file: <图片文件>
```

响应：
```json
{
  "success": true,
  "text": "翻译后的文本",
  "message": "翻译成功"
}
```

## 测试

### 使用测试脚本

```bash
# 健康检查
python test_api.py --mode health

# 测试视频翻译
python test_api.py --mode video --file /path/to/video.mp4

# 测试图片翻译
python test_api.py --mode image --file /path/to/image.jpg
```

### 使用 curl

```bash
# 视频翻译
curl -X POST "http://localhost:8000/translate/video" \
     -F "file=@/path/to/video.mp4"

# 图片翻译
curl -X POST "http://localhost:8000/translate/image" \
     -F "file=@/path/to/image.jpg"
```

### 使用 Python requests

```python
import requests

url = "http://localhost:8000/translate/video"
with open("video.mp4", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    result = response.json()
    print(result["text"])
```

## Docker 部署（可选）

创建 `Dockerfile`：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . .
RUN pip install -r api_requirements.txt

EXPOSE 8000

CMD ["bash", "start_api.sh"]
```

构建并运行：

```bash
docker build -t unisign-api .
docker run -p 8000:8000 --gpus all unisign-api
```

## 注意事项

1. **模型权重**: 确保 `CHECKPOINT_PATH` 指向正确的模型检查点文件
2. **GPU 内存**: 视频翻译需要较多显存，如显存不足可调整 `MAX_LENGTH` 或使用 CPU
3. **姿态提取**: 首次运行时会自动下载姿态提取模型
4. **支持格式**:
   - 视频: mp4, avi, mov, mkv, flv, wmv
   - 图片: jpg, jpeg, png, bmp, tiff, webp
