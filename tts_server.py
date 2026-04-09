"""
TTS 独立服务 - 运行在单独 venv 中
使用 PyTorch 2.2 + ChatTTS 0.2.4
"""
import os
import io
import wave
import tempfile
import datetime
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
import ChatTTS
import torch

app = FastAPI(title="语见 TTS Service")

# 全局 TTS 实例
chat = None
speaker_dir = "./speaker"
wavs_dir = "./static/wavs"

@app.on_event("startup")
async def startup():
    global chat
    os.makedirs(speaker_dir, exist_ok=True)
    os.makedirs(wavs_dir, exist_ok=True)

    print("[*] 加载 ChatTTS 模型...")
    chat = ChatTTS.Chat()
    chat.load(source="huggingface")
    print("[*] ChatTTS 模型加载完成!")

@app.post("/tts")
async def tts(
    text: str = Form(...),
    voice: str = Form("2222"),
    temperature: float = Form(0.3),
    top_p: float = Form(0.7),
    top_k: int = Form(20),
    speed: int = Form(5),
):
    """语音合成接口"""
    try:
        # 输入验证
        text = text.strip()
        if not text or len(text) < 1:
            return JSONResponse(status_code=400, content={"error": "文本不能为空"})

        # 加载音色
        voice = voice.replace('.csv', '.pt')
        seed_path = f'{speaker_dir}/{voice}'

        if voice.endswith('.pt') and os.path.exists(seed_path):
            rand_spk = torch.load(seed_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        else:
            voice_int = int(''.join(filter(str.isdigit, voice))) if any(c.isdigit() for c in voice) else 2222
            torch.manual_seed(voice_int)
            rand_spk = chat.sample_random_speaker()
            torch.save(rand_spk, f"{speaker_dir}/{voice_int}.pt")

        # 参数
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rand_spk,
            prompt=f"[speed_{speed}]",
            top_P=top_p,
            top_K=top_k,
            temperature=temperature,
        )

        # 合成
        wavs = chat.infer([text], params_infer_code=params_infer_code)

        # 保存
        filename = datetime.datetime.now().strftime('%H%M%S_') + f"{voice}.wav"
        filepath = os.path.join(wavs_dir, filename)

        if len(wavs) > 0:
            wav = wavs[0]
            wav_bytes = (wav * 32768).astype(np.int16).tobytes()
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(wav_bytes)

        return FileResponse(filepath, media_type="audio/wav", filename=filename)

    except Exception as e:
        print(f"[TTS ERROR] {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health():
    return {"status": "healthy", "tts_loaded": chat is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6009)
