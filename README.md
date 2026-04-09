<h1 align="center">语见 (YuJian)</h1>
<p align="center">
  <strong>面向聋哑人群体的智能手语翻译与心理支持系统</strong>
</p>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/PyTorch-2.8+-red.svg" alt="PyTorch">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/FastAPI-现代Web框架-green.svg" alt="FastAPI">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </a>
</p>

---

## 🎯 项目介绍

**语见**是一款专为聋哑人群体设计的智能服务系统，融合手语翻译、心理支持和语音合成技术，搭建聋哑人与外界沟通的桥梁。

> 💡 **为什么叫"语见"？**
> 
> "语"代表语言、沟通，"见"代表看见、理解。我们希望让每一份手语都能被"看见"，让每一个声音都能被"听见"。

---

## ✨ 核心功能

### 🤟 手语翻译
- **视频翻译**：上传手语视频，自动识别并转换为文字
- **图片翻译**：支持单张手语图片识别
- **姿态提取**：基于 RTMLib 的轻量级姿态估计
- **支持数据集**：CSL Daily、CSL News

### 🧠 智能心理支持 (EmoLLM)
- **心理健康咨询**：基于 EmoLLM 大模型的专业心理支持
- **RAG 知识增强**：融合心理学知识库，提供专业建议
- **手语友好设计**：理解手语语序特点，不纠正语法
- **温柔共情回复**：亲切、可爱的沟通风格

### 🔊 语音合成 (TTS)
- **ChatTTS 集成**：高质量中文语音合成
- **多种音色**：8+ 种预设音色可选
- **独立服务架构**：避免依赖冲突，稳定运行
- **自动转发**：主 API 端口统一对外服务

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        语见 API (端口 6006)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  手语翻译     │  │  EmoLLM 咨询  │  │   TTS 转发       │  │
│  │  (Uni-Sign)  │  │  (RAG + LLM) │  │  (→ 端口 6009)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   TTS 独立服务 (端口 6009)                   │
│              (PyTorch 2.2 + ChatTTS 隔离环境)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- CUDA 12.8+
- 显存：16GB+ (推荐 24GB)

### 安装步骤

1. **克隆仓库**
```bash
git clone git@github.com:Kirawii/Yujian.git
cd Yujian
```

2. **安装主环境依赖**
```bash
pip install -r requirements.txt
pip install -r api_requirements.txt
```

3. **配置 TTS 独立服务**
```bash
# 创建 TTS 虚拟环境
python -m venv tts_env
source tts_env/bin/activate

# 安装 TTS 依赖 (PyTorch 2.2 以避免冲突)
pip install torch==2.2.0 torchaudio==2.2.0
pip install ChatTTS fastapi uvicorn

deactivate
```

4. **下载模型权重**
```bash
# 手语翻译模型 (CSL Daily)
mkdir -p pretrained_weight
# 下载地址: https://huggingface.co/ZechengLi19/Uni-Sign

# EmoLLM 模型
# 下载地址: /home/xtk/models/haiyangpengai/careyou_7b_16bit_v3_2_qwen14_4bit
```

5. **启动服务**
```bash
# 一键启动所有服务
./start_all.sh

# 或手动分别启动
# 1. 启动 TTS 服务
source tts_env/bin/activate
uvicorn tts_server:app --host 0.0.0.0 --port 6009

# 2. 启动主 API
./start.sh
```

---

## 📡 API 接口

### 健康检查
```bash
curl http://localhost:6006/health
```

### 手语翻译 - 视频
```bash
curl -X POST "http://localhost:6006/translate/video" \
  -F "file=@example.mp4"
```

### 手语翻译 - 图片
```bash
curl -X POST "http://localhost:6006/translate/image" \
  -F "file=@example.jpg"
```

### 心理咨询对话
```bash
curl -X POST "http://localhost:6006/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "我最近感到很焦虑"}],
    "max_new_tokens": 512
  }'
```

### 语音合成 (TTS)
```bash
curl -X POST "http://localhost:6006/tts" \
  -F "text=你好，我是语见助手" \
  -F "voice=2222" \
  --output output.wav
```

📖 **完整 API 文档**: [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)

---

## 📁 项目结构

```
Yujian/
├── api.py                  # 主 API 服务
├── tts_server.py           # TTS 独立服务
├── models.py               # Uni-Sign 模型定义
├── datasets.py             # 数据集处理
├── config.py               # 配置文件
├── utils.py                # 工具函数
├── start.sh                # 启动主 API
├── start_all.sh            # 一键启动所有服务
├── stop.sh                 # 停止服务
├── status.sh               # 查看服务状态
├── tts_env/                # TTS 虚拟环境
├── pretrained_weight/      # 模型权重
├── speaker/                # 音色文件
├── static/wavs/            # 生成的音频
└── demo/                   # 演示代码
```

---

## 🛠️ 技术栈

| 模块 | 技术 |
|------|------|
| 手语翻译 | Uni-Sign、RTMLib、ST-GCN |
| 心理咨询 | EmoLLM (Qwen2)、RAG (FAISS)、LangChain |
| 语音合成 | ChatTTS、FastAPI |
| Web 框架 | FastAPI、Uvicorn |
| 深度学习 | PyTorch 2.8 (主) / 2.2 (TTS) |
| 向量数据库 | FAISS |
| 姿态估计 | RTMPose、YOLOX |

---

## 📝 配置说明

### 环境变量
```bash
# 模型路径
export CHECKPOINT_PATH="./pretrained_weight/best_model.pth"
export EMOLLM_MODEL_PATH="/path/to/careyou_7b"
export EMOLLM_TOKENIZER_PATH="/path/to/qwen_tokenizer"

# 设备配置
export DEVICE="cuda"
export ENABLE_TTS="true"
export ENABLE_EMOLLM="true"
```

### 端口配置
- `6006` - 主 API 服务（对外暴露）
- `6009` - TTS 独立服务（内部转发）

---

## 🎓 团队成员

本项目由以下成员共同开发：

| 姓名 | 职责 |
|------|------|
| 许桐恺 | 项目负责人、架构设计 |
| 刘犇 | 模型开发、训练优化 |
| 刘勇杰 | API 开发、部署运维 |
| 李家豪 | 前端对接、测试验证 |
| 任欣蕊 | 产品设计、文档编写 |

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢

本项目基于以下开源项目构建：

- [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) - 手语翻译基础模型
- [EmoLLM](https://github.com/EmoLLM) - 心理健康大模型
- [ChatTTS](https://github.com/2noise/ChatTTS) - 语音合成
- [RTMLib](https://github.com/Tau-J/rtmlib) - 姿态估计

感谢所有开源贡献者！

---

## 📮 联系我们

如有问题或建议，欢迎通过以下方式联系：

- 📧 Email: [待补充]
- 💬 Issue: [GitHub Issues](https://github.com/Kirawii/Yujian/issues)

---

<p align="center">
  <strong>让沟通无障碍，让心灵有依靠 ❤️</strong>
</p>
