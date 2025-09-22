from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.gen_tts import TTS
import torchaudio as ta
import io
import base64
from typing import Optional


class GenerateRequest(BaseModel):
    text: str
    language_id: Optional[str] = "ko"
    audio_prompt: Optional[str] = "miyako"
    cfg_weight: Optional[float] = 0.5
    exaggeration: Optional[float] = 0.5
    temperature: Optional[float] = 0.8
    gen_timestamp: Optional[bool] = True


class GenerateResponse(BaseModel):
    audio_wav_base64: str
    timestamps: list


app = FastAPI(title="TTS API")


@app.on_event("startup")
def startup_event():
    # 전역 TTS 인스턴스 생성
    global tts
    tts = TTS()


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        # 입력으로 audio_prompt가 들어오면 재초기화
        if getattr(tts, "ref_path", None) is None or req.audio_prompt:
            # 간단하게 새로운 TTS 인스턴스로 교체
            tts = TTS(audio_prompt=req.audio_prompt)

        result = tts.generate(
            req.text,
            language_id=req.language_id,
            cfg_weight=req.cfg_weight,
            exaggeration=req.exaggeration,
            temperature=req.temperature,
            gen_timestamp=req.gen_timestamp,
        )

        audio_tensor = result.get("audio")
        timestamps = result.get("timestamps", [])

        if audio_tensor is None:
            raise HTTPException(status_code=500, detail="오디오 생성 실패")

        # torchaudio expects shape [channels, time]
        if audio_tensor.dim() == 1:
            # mono: [time] -> [1, time]
            audio_tensor = audio_tensor.unsqueeze(0)

        buffer = io.BytesIO()
        ta.save(buffer, audio_tensor, tts.sr, format="wav")
        buffer.seek(0)
        wav_bytes = buffer.read()
        wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

        return GenerateResponse(audio_wav_base64=wav_b64, timestamps=timestamps)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)