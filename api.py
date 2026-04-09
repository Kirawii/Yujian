"""
语见 FastAPI 服务
支持图片/视频转文字（手语翻译）+ 语音合成
"""

import os
import sys
import io
import tempfile
import cv2
import numpy as np
import torch
import wave
import datetime
import time
import re
import subprocess
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from random import random

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# 导入 语见 相关模块
from models import Uni_Sign
from datasets import S2T_Dataset_online, load_part_kp
from config import *
import utils

# 尝试导入 rtmlib
try:
    from rtmlib import Wholebody
except ImportError:
    print("警告: rtmlib 未安装，姿态提取功能将不可用")
    Wholebody = None

# 导入 ChatTTS
try:
    import ChatTTS
    from ChatTTS.utils import select_device
    CHATTTS_AVAILABLE = True
except ImportError:
    print("警告: ChatTTS 未安装，语音合成功能将不可用")
    CHATTTS_AVAILABLE = False

# 导入 EmoLLM 所需库
try:
    # accelerate 必须在 transformers 之前导入，用于支持 device_map="auto"
    import importlib
    accelerate = importlib.import_module('accelerate')
    transformers = importlib.import_module('transformers')
    AutoTokenizer = transformers.AutoTokenizer
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    GenerationConfig = transformers.GenerationConfig
    EMOLLM_AVAILABLE = True
except ImportError as e:
    print(f"警告: transformers/accelerate 未安装 ({e})，EmoLLM 功能将不可用")
    EMOLLM_AVAILABLE = False

# 导入 RAG 相关库
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    RAG_AVAILABLE = True
except ImportError:
    print("警告: RAG 依赖未安装，知识库检索将不可用")
    RAG_AVAILABLE = False


class SignLanguageTranslator:
    """手语翻译器封装类"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        rgb_support: bool = False,
        max_length: int = 256,
        dataset: str = "CSL_Daily",
        target_fps: float = None,  # None表示不采样，使用所有帧；如5.0表示每秒采5帧
    ):
        self.device = device
        self.rgb_support = rgb_support
        self.max_length = max_length
        self.dataset = dataset
        self.checkpoint_path = checkpoint_path
        self.target_fps = float(target_fps) if target_fps else None

        # 创建参数对象
        self.args = self._create_args()

        # 加载模型
        self.model = self._load_model()

        # 初始化姿态提取器
        self.wholebody = None
        if Wholebody is not None:
            self.wholebody = Wholebody(
                to_openpose=False,
                mode="lightweight",
                backend="onnxruntime",
                device=device
            )

        print(f"模型加载完成，使用设备: {device}")

        # 预热：执行一次虚拟推理，将模型加载到GPU
        print("[INFO] 预热模型...")
        try:
            import numpy as np
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.wholebody(dummy_frame)
            print("[INFO] 预热完成")
        except Exception as e:
            print(f"[WARN] 预热失败: {e}")

    def _create_args(self):
        """创建参数对象"""
        class Args:
            pass

        args = Args()
        args.seed = 42
        args.batch_size = 1
        args.gradient_accumulation_steps = 8
        args.gradient_clipping = 1.0
        args.epochs = 20
        args.world_size = 1
        args.dist_url = 'env://'
        args.local_rank = 0
        args.hidden_dim = 256
        args.finetune = self.checkpoint_path
        args.opt = 'adamw'
        args.opt_eps = 1.0e-09
        args.opt_betas = None
        args.clip_grad = None
        args.momentum = 0.9
        args.weight_decay = 0.0001
        args.sched = 'cosine'
        args.lr = 1.0e-3
        args.min_lr = 1.0e-08
        args.warmup_epochs = 0
        args.output_dir = ''
        args.eval = False
        args.num_workers = 8
        args.pin_mem = True
        args.offload = False
        args.dtype = 'bf16'
        args.zero_stage = 2
        args.compute_fp32_loss = False
        args.quick_break = 0
        args.rgb_support = self.rgb_support
        args.max_length = self.max_length
        args.dataset = self.dataset
        args.task = "SLT"
        args.label_smoothing = 0.2
        args.online_video = ""

        return args

    def _load_model(self):
        """加载模型权重"""
        print("正在加载模型...")
        model = Uni_Sign(args=self.args)
        model.to(self.device)
        model.train()

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f'加载检查点: {self.checkpoint_path}')
            state_dict = torch.load(self.checkpoint_path, map_location='cpu')['model']
            ret = model.load_state_dict(state_dict, strict=True)
            if ret.missing_keys:
                print('Missing keys:', ret.missing_keys)
            if ret.unexpected_keys:
                print('Unexpected keys:', ret.unexpected_keys)
        else:
            print(f"警告: 检查点文件不存在 {self.checkpoint_path}")

        model.eval()
        if self.device == "cuda":
            model.to(torch.bfloat16)

        return model

    def extract_pose_from_video(self, video_path: str, max_workers: int = 16):
        """
        从视频中提取姿态关键点

        如果设置了 target_fps，会按目标帧率采样；否则读取所有帧
        """
        if self.wholebody is None:
            raise RuntimeError("rtmlib 未安装，无法提取姿态")

        data = {"keypoints": [], "scores": []}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频信息: {fps:.2f} FPS, 总帧数: {total_frames}")

        vid_data = []
        frame_indices = []

        if self.target_fps is not None and fps > 0:
            # 按目标FPS采样
            sampling_interval = fps / self.target_fps
            frame_idx = 0
            next_sample_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx >= next_sample_idx:
                    vid_data.append(frame)
                    frame_indices.append(frame_idx)
                    next_sample_idx += sampling_interval

                frame_idx += 1

            print(f"FPS采样: 目标 {self.target_fps} FPS, 采样间隔 {sampling_interval:.2f}, 采样帧数: {len(vid_data)}")
        else:
            # 读取所有帧
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                vid_data.append(frame)
            print(f"读取所有帧: {len(vid_data)} 帧")

        cap.release()

        if len(vid_data) == 0:
            raise ValueError("视频为空")

        def process_frame(frame):
            frame = np.uint8(frame)
            keypoints, scores = self.wholebody(frame)
            H, W, C = frame.shape
            return keypoints, scores, [W, H]

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_frame, frame) for frame in vid_data]
            for f in tqdm(futures, desc="处理视频帧", total=len(vid_data)):
                results.append(f.result())

        for keypoints, scores, w_h in results:
            data['keypoints'].append(keypoints / np.array(w_h)[None, None])
            data['scores'].append(scores)

        return data

    def extract_pose_from_image(self, image_path: str):
        """从单张图片中提取姿态关键点"""
        if self.wholebody is None:
            raise RuntimeError("rtmlib 未安装，无法提取姿态")

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"无法读取图片: {image_path}")

        frame = np.uint8(frame)
        keypoints, scores = self.wholebody(frame)
        H, W, C = frame.shape

        data = {
            "keypoints": [keypoints / np.array([W, H])[None, None]],
            "scores": [scores]
        }

        return data

    def translate(self, pose_data: dict, video_path: str = None) -> str:
        """执行手语翻译"""
        # 创建在线数据集
        online_data = S2T_Dataset_online(args=self.args)
        online_data.pose_data = pose_data

        # RGB 模式需要设置视频路径
        if self.rgb_support and video_path:
            online_data.rgb_data = video_path

        # 创建数据加载器
        online_sampler = torch.utils.data.SequentialSampler(online_data)
        online_dataloader = torch.utils.data.DataLoader(
            online_data,
            batch_size=1,
            collate_fn=online_data.collate_fn,
            sampler=online_sampler,
        )

        # 推理
        self.model.eval()
        target_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        with torch.no_grad():
            tgt_pres = []

            for step, (src_input, tgt_input) in enumerate(online_dataloader):
                if target_dtype is not None:
                    for key in src_input.keys():
                        if isinstance(src_input[key], torch.Tensor):
                            src_input[key] = src_input[key].to(target_dtype).to(self.device)

                stack_out = self.model(src_input, tgt_input)
                output = self.model.generate(
                    stack_out,
                    max_new_tokens=100,
                    num_beams=4,
                )

                for i in range(len(output)):
                    tgt_pres.append(output[i])

            # 解码结果
            tokenizer = self.model.mt5_tokenizer
            padding_value = tokenizer.eos_token_id

            pad_tensor = torch.ones(150 - len(tgt_pres[0])).to(self.device) * padding_value
            tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

            from torch.nn.utils.rnn import pad_sequence
            tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
            tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

            result = tgt_pres[0] if tgt_pres else ""

            # 清理 MT5 特殊 token (如 <extra_id_0>)
            import re
            result = re.sub(r'<extra_id_\d+>', '', result)
            result = result.strip()

            return result


class TTSEngine:
    """语音合成引擎封装类"""

    def __init__(self, device: str = "cuda"):
        self.device_str = device
        self.chat = None
        self.speaker_dir = "./speaker"
        self.wavs_dir = "./static/wavs"
        self.device = None

        # 创建音频存储目录
        os.makedirs(self.wavs_dir, exist_ok=True)
        os.makedirs(self.speaker_dir, exist_ok=True)

        if CHATTTS_AVAILABLE:
            self._load_model()
        else:
            print("警告: ChatTTS 不可用，语音合成功能将无法使用")

    def _load_model(self):
        """加载 ChatTTS 模型"""
        print("正在加载 ChatTTS 模型...")

        # 选择设备
        if self.device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.chat = ChatTTS.Chat()

        # 加载模型 - 使用 PyPI 版本的 API
        try:
            # 尝试从本地加载
            if os.path.exists("models/pzc163/chattts"):
                self.chat.load(source="custom", custom_path='./models/pzc163/chattts', device=self.device)
            else:
                # 从 HuggingFace 下载
                print("正在从 HuggingFace 下载 ChatTTS 模型...")
                self.chat.load(source="huggingface", device=self.device)
            print("ChatTTS 模型加载完成")
        except Exception as e:
            print(f"ChatTTS 模型加载失败: {e}")
            self.chat = None
            raise

    def synthesize(
        self,
        text: str,
        voice: str = "2222",
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
        speed: int = 5,
        skip_refine: bool = False
    ) -> str:
        """
        合成语音

        参数:
            text: 要合成的文本
            voice: 音色ID或音色文件(.pt)
            temperature: 采样温度
            top_p: top-p 采样
            top_k: top-k 采样
            speed: 语速 (0-10)
            skip_refine: 是否跳过文本优化

        返回:
            生成的音频文件路径
        """
        if self.chat is None:
            raise RuntimeError("ChatTTS 模型未加载")

        # 输入验证和清洗
        if not text:
            raise ValueError("文本不能为空")

        # 清洗文本：移除多余空白、控制字符
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # 多个空白合并为单个空格
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)  # 移除控制字符

        # 检查清洗后文本
        if not text or len(text) < 2:
            raise ValueError("文本过短或只包含特殊字符，无法合成")

        if len(text) > 1000:
            raise ValueError("文本过长（超过1000字），请分段合成")
        # 短文本（<=15字）启用 refine_text，长文本跳过以提高速度
        should_refine = len(text) <= 15

        # 加载音色
        rand_spk = self._load_speaker(voice)

        # 准备参数
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rand_spk,
            prompt=f"[speed_{speed}]",
            top_P=top_p,
            top_K=top_k,
            temperature=temperature,
            max_new_token=2048
        )

        params_refine_text = ChatTTS.Chat.RefineTextParams(
            top_P=top_p,
            top_K=top_k,
            temperature=temperature,
            max_new_token=384
        )

        # 执行推理
        start_time = time.time()

        # 执行推理 - 简化参数避免兼容性问题
        wavs = self.chat.infer([text])

        inference_time = round(time.time() - start_time, 2)

        # 保存音频文件
        filename = datetime.datetime.now().strftime('%H%M%S_') + \
                   f"use{inference_time}s-seed{voice}-te{temperature}-tp{top_p}-{str(random())[2:7]}.wav"
        filepath = os.path.join(self.wavs_dir, filename)

        # 保存为 WAV 格式
        if len(wavs) > 0:
            wav = wavs[0]
            # 使用 wave 模块保存
            wav_bytes = (wav * 32768).astype(np.int16).tobytes()
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(wav_bytes)

        return filepath

    def _load_speaker(self, voice: str):
        """加载或生成音色"""
        voice = voice.replace('.csv', '.pt')
        seed_path = f'{self.speaker_dir}/{voice}'

        # 尝试加载音色文件
        if voice.endswith('.pt') and os.path.exists(seed_path):
            print(f'使用音色: {seed_path}')
            return torch.load(seed_path, map_location=self.device)

        # 根据 seed 生成随机音色
        print(f'生成随机音色: seed={voice}')
        voice_int = re.findall(r'^(\d+)', voice)
        if len(voice_int) > 0:
            voice_seed = int(voice_int[0])
        else:
            voice_seed = 2222

        torch.manual_seed(voice_seed)
        rand_spk = self.chat.sample_random_speaker()

        # 保存音色供下次使用
        torch.save(rand_spk, f"{self.speaker_dir}/{voice_seed}.pt")

        return rand_spk


class EmoLLMEngine:
    """EmoLLM 心理健康咨询模型封装类 (DeepSeek-R1/Qwen 版本)"""

    def __init__(
        self,
        model_path: str = "/home/xtk/models/haiyangpengai/careyou_7b_16bit_v3_2_qwen14_4bit",
        tokenizer_path: str = "/home/xtk/qwen_tokenizer/qwen/Qwen2-7B-Instruct",
        device: str = "cuda",
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.device_str = device
        self.temperature = temperature
        self.top_p = top_p
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.model = None
        self.tokenizer = None

        # RAG 相关
        self.rag_enabled = True  # 默认启用 RAG
        self.vectorstore = None
        self.embeddings = None

        if EMOLLM_AVAILABLE:
            self._load_model()
            self._load_rag()  # 加载 RAG
        else:
            print("警告: EmoLLM 依赖不可用")

    def _load_model(self):
        """加载 EmoLLM 模型"""
        print(f"[*] 正在加载 EmoLLM 模型: {self.model_path}")
        print(f"[*] 使用 Tokenizer: {self.tokenizer_path}")
        print(f"[*] GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

        try:
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载半精度模型到指定设备
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device_str)
            self.model.eval()

            print(f"[*] 模型设备: {next(self.model.parameters()).device}")
            print("[*] EmoLLM 模型加载完成!")

        except Exception as e:
            print(f"[*] EmoLLM 模型加载失败: {e}")

    def _load_rag(self):
        """加载 RAG 知识库"""
        if not RAG_AVAILABLE:
            print("[*] RAG 不可用，跳过知识库加载")
            return

        # Vector DB 路径
        vector_db_paths = [
            "/root/autodl-tmp/liuyongjie/Uni-Sign/EmoLLM/EmoLLM/careyou/EmoLLMRAGTXT/vector_db",
            "./EmoLLM/EmoLLM/careyou/EmoLLMRAGTXT/vector_db",
        ]

        # 查找存在的 Vector DB
        vector_db_path = None
        for path in vector_db_paths:
            if os.path.exists(path):
                vector_db_path = path
                break

        if not vector_db_path:
            print("[*] 未找到 Vector DB，RAG 功能将不可用")
            return

        try:
            print(f"[*] 正在加载 RAG 知识库: {vector_db_path}")

            # 加载 Embedding 模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={"device": self.device_str},
                encode_kwargs={"normalize_embeddings": True}
            )

            # 加载 Vector DB
            self.vectorstore = FAISS.load_local(
                vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            print("[*] RAG 知识库加载完成!")

        except Exception as e:
            print(f"[*] RAG 知识库加载失败: {e}")
            self.vectorstore = None

    def _retrieve_knowledge(self, query: str, k: int = 3) -> str:
        """检索相关知识"""
        if not self.rag_enabled or self.vectorstore is None:
            return ""

        try:
            # 相似度搜索
            docs = self.vectorstore.similarity_search(query, k=k)

            # 合并检索结果
            knowledge = "\n\n".join([doc.page_content for doc in docs])

            return knowledge

        except Exception as e:
            print(f"[*] 知识检索失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _build_prompt(self, messages: list, knowledge: str = "") -> str:
        """构建 DeepSeek-R1 格式的 prompt，添加中文系统提示和检索知识"""
        prompt_parts = []

        # 基础系统提示词
        base_system = """你是一个由计算机设计大赛许桐恺、刘犇、刘勇杰、李家豪、任欣蕊研发的心理健康大模型，专门为聋哑人群体提供心理支持与帮助。

我会向你表达一些情绪或心理问题，你需要用专业的心理学知识，帮助我理解和缓解这些问题。

在回答时，请遵循以下要求：

【沟通风格】

使用温柔、亲切、可爱、略带俏皮的语气 😊
多表达理解、共情和支持，让用户感到被关心和陪伴
避免说教或批评，语气自然、像真实的人在交流
可以适当使用 Emoji 或简单表情（但不要过多）

【表达方式（重点）】

使用简单、清晰、容易理解的句子
尽量使用常见词汇，避免专业术语或复杂表达
句子尽量短，语序自然、直白
可以分点或换行，让内容更容易阅读
每次回答聚焦 1–3 个重点，不要信息过多

【针对聋哑用户的理解（非常重要）】

用户的表达可能受到手语影响，出现语序不标准、句子不完整或表达跳跃的情况
你需要优先理解"想表达的意思"，而不是字面语序
在内部自动将用户的话整理为通顺含义后再回答
不要纠正语法，不要指出用户表达有问题
不要评价用户的表达方式

【澄清与引导】

如果理解不清，可以用简单、温柔的问题确认
（例如："你的意思是这个吗？"、"我理解对吗？"）
通过一步步引导，帮助用户更清楚表达自己

【专业支持】

在温柔表达的同时，提供有依据的心理学建议
优先帮助用户识别情绪、理解原因、找到简单可行的应对方法
不做武断诊断

请始终记住：
👉 目标不是纠正用户，而是理解用户、支持用户、陪伴用户 🌱"""

        # 融入检索到的知识
        if knowledge:
            knowledge_section = f"""

【参考知识库】
以下是从心理学知识库中检索到的相关心理学理论、方法论和参考案例，供你参考：
{knowledge}

【重要提醒】
- 上述知识库内容只是参考资料，不是用户的情况描述
- 不要假设用户的具体情况（如分手、家庭矛盾等），这些可能来自案例库而非当前用户
- 必须通过提问来了解用户的真实情况，不要替用户编造故事
- 回答时聚焦用户的实际输入，结合知识库的专业方法提供支持"""
            base_system += knowledge_section

        system_msg = base_system

        # 查找用户自定义的系统提示
        for msg in messages:
            if msg.get("role") == "system":
                user_system = msg.get("content", "")
                # 在自定义提示后追加知识库（加上重要提醒）
                if knowledge:
                    user_system += f"\n\n【参考知识】\n{knowledge}\n\n【重要提醒】知识库只是参考资料，不是用户的情况描述，不要假设用户具体情况，必须通过提问了解用户真实情况。"
                system_msg = user_system
                break

        # 添加系统提示
        if system_msg:
            prompt_parts.append(f"<｜User｜>{system_msg}<｜Assistant｜>好的，我会用中文回答您的问题。<｜end▁of▁sentence｜>")

        # 对话历史 - 保留完整的 assistant 回复（包括 think 标签）
        for msg in messages:
            if msg.get("role") == "user":
                prompt_parts.append(f"<｜User｜>{msg.get('content', '')}")
            elif msg.get("role") == "assistant":
                content = msg.get("content", "")
                prompt_parts.append(f"<｜Assistant｜>{content}<｜end▁of▁sentence｜>")

        # 添加 assistant 开头
        prompt_parts.append("<｜Assistant｜>")

        return "".join(prompt_parts)

    @torch.inference_mode()
    def chat(
        self,
        messages: list,
        max_new_tokens: int = 512,
        temperature: float = None,
        top_p: float = None,
    ) -> str:
        """
        进行心理咨询对话

        参数:
            messages: 对话历史，格式 [{"role": "user"/"assistant"/"system", "content": "..."}]
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: top-p 采样
        返回:
            生成的回复文本
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("EmoLLM 模型未加载")

        # 获取用户最新输入，用于 RAG 检索
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        # RAG 检索知识（默认启用）
        knowledge = ""
        if self.rag_enabled and user_query:
            knowledge = self._retrieve_knowledge(user_query)
            if knowledge:
                print(f"[*] RAG 检索到 {len(knowledge)} 字符相关知识")

        # 构建 prompt（融入检索知识）
        text = self._build_prompt(messages, knowledge)

        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        input_length = input_ids.shape[1]

        # 生成参数
        temp = temperature if temperature is not None else self.temperature
        tp = top_p if top_p is not None else self.top_p

        # 执行生成 - 使用 StoppingCriteria 在特定标记处停止
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnTokenSequences(StoppingCriteria):
            def __init__(self, tokenizer, stop_strings):
                self.stop_token_ids = [
                    tokenizer.encode(s, add_special_tokens=False)
                    for s in stop_strings
                ]

            def __call__(self, input_ids, scores, **kwargs):
                for stop_ids in self.stop_token_ids:
                    if len(input_ids[0]) < len(stop_ids):
                        continue
                    if input_ids[0][-len(stop_ids):].tolist() == stop_ids:
                        return True
                return False

        stopper = StopOnTokenSequences(self.tokenizer, ["<｜User｜>", "<｜Assistant｜>"])

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=tp,
            repetition_penalty=1.1,
            eos_token_id=151643,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )

        # 解码输出
        new_tokens = outputs[0][input_length:]
        raw_response = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        # 清理响应格式
        response = raw_response

        # 1. 找到下一个特殊标记的位置并截断（防止模型生成用户消息）
        stop_tokens = ["<｜User｜>", "<｜Assistant｜>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        for token in stop_tokens:
            if token in response:
                response = response.split(token)[0]

        # 2. 去重：移除连续的重复段落
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = None
        for line in lines:
            # 跳过完全相同的连续行
            if line.strip() and line.strip() == prev_line:
                continue
            cleaned_lines.append(line)
            prev_line = line.strip() if line.strip() else prev_line

        response = '\n'.join(cleaned_lines).strip()

        return response


# 创建 FastAPI 应用
app = FastAPI(
    title="语见 手语翻译服务",
    description="支持图片/视频转文字的手语翻译 API",
    version="1.0.0"
)

# 全局翻译器实例
translator: Optional[SignLanguageTranslator] = None

# 全局语音合成引擎实例
tts_engine: Optional[TTSEngine] = None

# 全局 EmoLLM 引擎实例
emollm_engine: Optional[EmoLLMEngine] = None


class TranslateResponse(BaseModel):
    success: bool
    text: str
    message: str


class TTSResponse(BaseModel):
    success: bool
    audio_url: str
    message: str
    inference_time: float


class TranslateAndSpeakResponse(BaseModel):
    success: bool
    text: str
    audio_url: str
    message: str


class ChatRequest(BaseModel):
    messages: list
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class ChatResponse(BaseModel):
    response: str
    status: str = "ok"


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global translator, tts_engine

    # 从环境变量获取配置
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "./pretrained_weight/best_model.pth")
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    rgb_support = os.environ.get("RGB_SUPPORT", "false").lower() == "true"
    max_length = int(os.environ.get("MAX_LENGTH", "256"))
    dataset = os.environ.get("DATASET", "CSL_Daily")
    target_fps = os.environ.get("TARGET_FPS")
    enable_tts = os.environ.get("ENABLE_TTS", "true").lower() == "true"
    if target_fps:
        target_fps = float(target_fps)

    print(f"初始化翻译器...")
    print(f"设备: {device}")
    print(f"RGB支持: {rgb_support}")
    print(f"数据集: {dataset}")
    print(f"目标FPS: {target_fps if target_fps else '使用所有帧 (无采样)'}")
    print(f"启用TTS: {enable_tts}")

    try:
        translator = SignLanguageTranslator(
            checkpoint_path=checkpoint_path,
            device=device,
            rgb_support=rgb_support,
            max_length=max_length,
            dataset=dataset,
            target_fps=target_fps,
        )
        print("翻译器初始化完成")
    except Exception as e:
        print(f"翻译器初始化失败: {e}")
        raise

    # 初始化语音合成引擎
    if enable_tts and CHATTTS_AVAILABLE:
        try:
            tts_engine = TTSEngine(device=device)
            print("语音合成引擎初始化完成")
        except Exception as e:
            print(f"语音合成引擎初始化失败: {e}")
            tts_engine = None
    else:
        print("语音合成引擎已跳过")

    # 初始化 EmoLLM 引擎
    global emollm_engine
    enable_emollm = os.environ.get("ENABLE_EMOLLM", "true").lower() == "true"
    emollm_model_path = os.environ.get("EMOLLM_MODEL_PATH", "/home/xtk/models/haiyangpengai/careyou_7b_16bit_v3_2_qwen14_4bit")
    emollm_tokenizer_path = os.environ.get("EMOLLM_TOKENIZER_PATH", "/home/xtk/qwen_tokenizer/qwen/Qwen2-7B-Instruct")

    print(f"启用EmoLLM: {enable_emollm}")

    if enable_emollm and EMOLLM_AVAILABLE:
        try:
            emollm_engine = EmoLLMEngine(
                model_path=emollm_model_path,
                tokenizer_path=emollm_tokenizer_path,
                device=device,
            )
            print("EmoLLM 引擎初始化完成")
        except Exception as e:
            print(f"EmoLLM 引擎初始化失败: {e}")
            emollm_engine = None
    else:
        print("EmoLLM 引擎已跳过")


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "语见 手语翻译 + EmoLLM 心理咨询服务",
        "version": "2.0.0",
        "endpoints": [
            "/translate/video - 上传视频进行手语翻译",
            "/translate/image - 上传图片进行手语翻译",
            "/tts - 文本转语音",
            "/tts/voices - 获取音色列表",
            "/chat - EmoLLM 心理咨询对话"
        ]
    }


@app.post("/translate/video", response_model=TranslateResponse)
async def translate_video(
    file: UploadFile = File(..., description="视频文件（mp4/avi/mov等格式）"),
):
    """
    上传视频文件进行手语翻译

    - **file**: 视频文件，支持 mp4, avi, mov 等常见格式
    """
    if translator is None:
        return TranslateResponse(
            success=False,
            text="",
            message="翻译器未初始化"
        )

    # 检查文件类型
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return TranslateResponse(
            success=False,
            text="",
            message=f"不支持的文件格式: {file_ext}，请上传视频文件"
        )

    try:
        # 保存上传的文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # 提取姿态
            print(f"正在提取视频姿态: {file.filename}")
            pose_data = translator.extract_pose_from_video(tmp_path)

            # 执行翻译（传入视频路径以支持RGB模式）
            print("正在执行翻译...")
            result_text = translator.translate(pose_data, video_path=tmp_path)

            return TranslateResponse(
                success=True,
                text=result_text,
                message="翻译成功"
            )
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        print(f"翻译失败: {e}")
        import traceback
        traceback.print_exc()
        return TranslateResponse(
            success=False,
            text="",
            message=f"翻译失败: {str(e)}"
        )


@app.post("/translate/image", response_model=TranslateResponse)
async def translate_image(
    file: UploadFile = File(..., description="图片文件（jpg/png等格式）"),
):
    """
    上传图片进行手语翻译（单帧手语识别）

    - **file**: 图片文件，支持 jpg, jpeg, png 等常见格式
    """
    if translator is None:
        return TranslateResponse(
            success=False,
            text="",
            message="翻译器未初始化"
        )

    # 检查文件类型
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return TranslateResponse(
            success=False,
            text="",
            message=f"不支持的文件格式: {file_ext}，请上传图片文件"
        )

    try:
        # 保存上传的文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # 提取姿态
            print(f"正在提取图片姿态: {file.filename}")
            pose_data = translator.extract_pose_from_image(tmp_path)

            # 执行翻译（传入视频路径以支持RGB模式）
            print("正在执行翻译...")
            result_text = translator.translate(pose_data, video_path=tmp_path)

            return TranslateResponse(
                success=True,
                text=result_text,
                message="翻译成功"
            )
        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        print(f"翻译失败: {e}")
        import traceback
        traceback.print_exc()
        return TranslateResponse(
            success=False,
            text="",
            message=f"翻译失败: {str(e)}"
        )


@app.post("/tts")
async def text_to_speech(
    text: str = Form(..., description="要合成的文本"),
    voice: str = Form("2222", description="音色ID (如 2222, 3333) 或音色文件名"),
    temperature: float = Form(0.3, description="采样温度 (0.1-1.0)"),
    top_p: float = Form(0.7, description="top-p 采样 (0.1-1.0)"),
    top_k: int = Form(20, description="top-k 采样 (1-100)"),
    speed: int = Form(5, description="语速 (0-10)"),
    skip_refine: bool = Form(False, description="是否跳过文本优化")
):
    """
    文本转语音

    - **text**: 要合成的文本 (必需)
    - **voice**: 音色ID，如 2222, 3333，或 .pt 音色文件 (默认: 2222)
    - **temperature**: 采样温度，影响语音多样性 (默认: 0.3)
    - **top_p**: nucleus sampling 参数 (默认: 0.7)
    - **top_k**: top-k sampling 参数 (默认: 20)
    - **speed**: 语速，0-10 (默认: 5)
    - **skip_refine**: 是否跳过文本优化阶段 (默认: False)
    """
    # 转发到 TTS 独立服务
    import httpx
    from fastapi.responses import StreamingResponse

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://localhost:6009/tts",
                data={
                    "text": text,
                    "voice": voice,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "speed": speed,
                }
            )

            if response.status_code == 200:
                # 流式返回音频
                from io import BytesIO
                return StreamingResponse(
                    BytesIO(response.content),
                    media_type="audio/wav",
                    headers={"Content-Disposition": f"attachment; filename=tts_{voice}.wav"}
                )
            else:
                return JSONResponse(
                    status_code=response.status_code,
                    content={"success": False, "message": response.text}
                )
    except Exception as e:
        print(f"[TTS PROXY ERROR] {e}")
        return JSONResponse(
            status_code=503,
            content={"success": False, "message": f"TTS 服务不可用: {str(e)}"}
        )


@app.get("/tts/voices")
async def list_voices():
    """获取可用音色列表"""
    voices = []

    # 预设音色
    preset_voices = [
        {"id": "2222", "name": "默认音色", "description": "温暖女声"},
        {"id": "3333", "name": "活泼音色", "description": "年轻女声"},
        {"id": "4444", "name": "沉稳音色", "description": "成熟男声"},
        {"id": "5555", "name": "清脆音色", "description": "清亮女声"},
        {"id": "6666", "name": "磁性音色", "description": "低沉男声"},
        {"id": "7777", "name": "温柔音色", "description": "柔和女声"},
        {"id": "8888", "name": "阳光音色", "description": "活力男声"},
        {"id": "9999", "name": "甜美音色", "description": "甜美女声"},
    ]

    # 检查speaker目录中的音色文件
    speaker_dir = "./speaker"
    if os.path.exists(speaker_dir):
        for file in os.listdir(speaker_dir):
            if file.endswith('.pt'):
                voice_id = file.replace('.pt', '')
                if voice_id not in [v["id"] for v in preset_voices]:
                    voices.append({
                        "id": voice_id,
                        "name": f"自定义音色 {voice_id}",
                        "description": "用户自定义音色"
                    })

    voices = preset_voices + voices

    return {
        "voices": voices,
        "count": len(voices)
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    EmoLLM 心理咨询对话接口

    - **messages**: 对话历史，格式 [{"role": "user"/"assistant"/"system", "content": "..."}]
    - **max_new_tokens**: 最大生成token数 (默认: 512)
    - **temperature**: 采样温度 (默认: 0.7)
    - **top_p**: top-p 采样 (默认: 0.9)

    **请求示例**:
    ```json
    {
        "messages": [
            {"role": "system", "content": "你是一个温柔的心理咨询师"},
            {"role": "user", "content": "我最近感到很焦虑"}
        ],
        "max_new_tokens": 512,
        "temperature": 0.7
    }
    ```
    """
    global emollm_engine

    if emollm_engine is None:
        return JSONResponse(
            status_code=503,
            content={"response": "EmoLLM 引擎未初始化", "status": "error"}
        )

    try:
        response = emollm_engine.chat(
            messages=request.messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return {"response": response, "status": "ok"}
    except Exception as e:
        import traceback
        print(f"EmoLLM 对话失败: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"response": f"Error: {str(e)}", "status": "error"}
        )


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": translator is not None,
        "tts_available": tts_engine is not None,
        "emollm_available": emollm_engine is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
