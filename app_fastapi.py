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
tts = TTS()

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
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
    uvicorn.run(app, host="127.0.0.1", port=14450)
