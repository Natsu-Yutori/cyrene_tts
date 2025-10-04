import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import torch
import logging
from g2pk import G2p
from pathlib import Path
import tempfile
from src.gen_timestamp import TTSTimestamp
from src.post_process import AudioPostProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seed = 123
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


class TTS:
    def __init__(self, audio_prompt="miyako"):
        self.g2p = G2p()
        self.model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        self.tts_timestamp = TTSTimestamp(whisper_model="small")
        self.sr = 24000
        self.post_processor = AudioPostProcessor(sample_rate=self.sr)

        match audio_prompt:
            case "miyako":
                self.ref_path = "audio_ref/miyako.wav"
            case "moon":
                self.ref_path = "audio_ref/moon.wav"
            case _:
                logger.warning("알 수 없는 오디오 프롬프트 입니다. 'miyako.wav'를 사용합니다.")
                self.ref_path = "audio_ref/miyako.wav"

    def set_seed(self, seed_value: int):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)
    
    def generate(self, text, language_id="ko", cfg_weight=0.5, exaggeration=0.5, temperature=0.8, gen_timestamp=True):
        result = {
            "audio": None,
            "timestamps": None
        }
        processed_text = " ".join(str(text).split())
        original_words = processed_text.split()

        # 음성 생성은 기존대로 phonemes를 사용
        phonemes = self._text_to_phonemes(text)
        audio_tensor = self._generate_speech(
            phonemes,
            language_id=language_id,
            audio_prompt_path=self.ref_path,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature
        )

        audio_tensor = self.post_processor.apply(audio_tensor, sample_rate=self.sr)

        if gen_timestamp:
            # 빈 문자열(단어 없음) 처리
            if not original_words:
                result["timestamps"] = []
            else:
                audio_duration = audio_tensor.shape[-1] / float(self.sr)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    ta.save(tmp_path, audio_tensor, self.sr)
                try:
                    # 4. Whisper로 초기 타임스탬프 추출
                    whisper_timestamps = self.tts_timestamp._extract_whisper_timestamps(
                        tmp_path,
                        processed_text,
                    )

                    # 5. 원본 텍스트와 정렬 및 연속 타임스탬프 생성
                    continuous_timestamps = self.tts_timestamp._create_continuous_timestamps(
                        original_words,
                        whisper_timestamps,
                        audio_duration
                    )
                finally:
                    # 임시 파일 삭제
                    Path(tmp_path).unlink(missing_ok=True)
                result["timestamps"] = continuous_timestamps
        result["audio"] = audio_tensor
        return result

    def _text_to_phonemes(self, text):
        return self.g2p(text)

    def _generate_speech(self, phonemes, language_id="ko", audio_prompt_path="miyako.wav", cfg_weight=0.5, exaggeration=0.5, temperature=0.8):
        return self.model.generate(phonemes, language_id=language_id, audio_prompt_path=audio_prompt_path, cfg_weight=cfg_weight, exaggeration=exaggeration, temperature=temperature)

