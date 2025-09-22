import asyncio
import io
import base64
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.gan import PerfectAlignmentTTS  # 클래스를 별도 파일로 저장했다고 가정


def _set_seed(seed: int = 80):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_set_seed()

# ======================== Request/Response Models ========================

class TTSRequest(BaseModel):
    """TTS 요청 모델"""
    text: str = Field(..., min_length=1, max_length=5000, description="변환할 텍스트")
    language_id: str = Field(default="ko", description="언어 코드")

class TTSResponse(BaseModel):
    """TTS 응답 모델"""
    audio_base64: str = Field(..., description="Base64 인코딩된 WAV 오디오")
    timestamps: List[Dict] = Field(..., description="단어별 타임스탬프")

# ======================== Model Manager ========================

class ModelManager:
    """모델 자원 관리"""
    
    def __init__(self, idle_timeout_minutes: int = 1):
        self.tts_system = None
        self.last_used = None
        self.idle_timeout = timedelta(minutes=idle_timeout_minutes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 동시 요청을 순차 처리하기 위한 전역 락
        self._lock = asyncio.Lock()
        
    def load_model(self):
        """모델 로드"""
        if self.tts_system is None:
            print(f"Loading TTS model on {self.device}...")
            self.tts_system = PerfectAlignmentTTS(
                device=self.device,
                whisper_model="small"
            )
            print("Model loaded successfully")
        self.last_used = datetime.now()
    
    def unload_model(self):
        """모델 언로드"""
        if self.tts_system is not None:
            print("Unloading model...")
            del self.tts_system
            self.tts_system = None
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            print("Model unloaded")
    
    def is_idle(self) -> bool:
        """유휴 상태 확인"""
        if self.last_used is None:
            return True
        return datetime.now() - self.last_used > self.idle_timeout
    
    async def generate(self, text: str, language_id: str) -> Tuple[bytes, List[Dict]]:
        """TTS 생성 (동시성 제어로 순차 처리)"""
        # 하나의 요청만 진입하도록 직렬화
        async with self._lock:
            # 모델 로드 확인 (경쟁 방지)
            if self.tts_system is None:
                self.load_model()

            self.last_used = datetime.now()

            # 비동기 실행 (CPU/GPU 작업은 스레드풀로 오프로딩)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self._generate_sync,
                text,
                language_id
            )
    
    def _generate_sync(self, text: str, language_id: str) -> Tuple[bytes, List[Dict]]:
        """동기 TTS 생성"""
        import torchaudio as ta
        
        # TTS 생성
        audio_tensor, timestamps = self.tts_system.generate(text, language_id)
        
        # WAV 바이트로 변환
        buffer = io.BytesIO()
        ta.save(buffer, audio_tensor, 24000, format="wav")
        audio_bytes = buffer.getvalue()
        buffer.close()

        return audio_bytes, timestamps

# ======================== Background Task ========================

async def cleanup_idle_models(model_manager: ModelManager):
    """유휴 모델 정리"""
    while True:
        await asyncio.sleep(60)  # 1분마다 체크
        
        # 언로드도 직렬화하여 경합 방지
        async with model_manager._lock:
            if model_manager.is_idle() and model_manager.tts_system is not None:
                model_manager.unload_model()

# ======================== FastAPI App ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기 관리"""
    # 시작
    print("Starting TTS API server...")
    cleanup_task = asyncio.create_task(cleanup_idle_models(app.state.model_manager))
    
    yield
    
    # 종료
    print("Shutting down TTS API server...")
    cleanup_task.cancel()
    if app.state.model_manager.tts_system is not None:
        app.state.model_manager.unload_model()

app = FastAPI(
    title="TTS API",
    description="TTS with perfect timestamps",
    version="1.0.0",
    lifespan=lifespan
)

# Model Manager 초기화
app.state.model_manager = ModelManager(idle_timeout_minutes=1)

# ======================== API Endpoint ========================

@app.post("/generate", response_model=TTSResponse)
async def generate_tts(request: TTSRequest):
    """
    TTS 생성 - WAV 오디오와 타임스탬프 JSON 반환
    """
    try:
        # TTS 생성
        audio_bytes, timestamps = await app.state.model_manager.generate(
            request.text,
            request.language_id
        )
        
        # Base64 인코딩
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # 응답
        return TTSResponse(
            audio_base64=audio_base64,
            timestamps=timestamps
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================== Run Server ========================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=4885,
        log_level="info"
    )
