import torch
import torchaudio as ta
import numpy as np
import whisper
from typing import List, Tuple, Dict
from pathlib import Path
import tempfile
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from difflib import SequenceMatcher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTSTimestamp:
    """텍스트와 완벽히 일치하는 연속 타임스탬프를 생성하는 TTS 시스템"""
    
    def __init__(self, whisper_model: str = "small"):
        """
        Args:
            whisper_model: whisper 모델 크기 ('tiny', 'base', 'small', 'medium', 'large')
        """
        
        # TTS 모델 초기화
        logger.info("Loading TTS model...")
        self.tts_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        
        # Whisper 모델 초기화
        logger.info(f"Loading Whisper {whisper_model} model...")
        self.whisper_model = whisper.load_model(whisper_model)
    
    def generate(
        self, 
        text: str, 
        language_id: str = "ko"
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        텍스트를 음성으로 변환하고 연속된 타임스탬프를 생성
        
        Args:
            text: 변환할 텍스트
            language_id: 언어 코드 ('ko', 'en', 등)
            
        Returns:
            (audio_tensor, timestamps): 오디오 텐서와 타임스탬프 딕셔너리 리스트
            timestamps 형식: [{"word": "안녕", "start": 0.0, "end": 0.5}, ...]
        """
        
        # 1. 텍스트 전처리 및 단어 분리
        processed_text = " ".join(text.split())  # 불필요한 공백 제거
        original_words = processed_text.split()
        
        if not original_words:
            return torch.zeros(1, 24000), []

        # 2. TTS로 음성 생성
        logger.info("Generating speech...")
        audio_tensor = self.tts_model.generate(processed_text, language_id=language_id, audio_prompt_path="moon.wav", cfg_weight=0.3, exaggeration=0.4, temperature=0.5)
        
        # 오디오 길이 (초)
        audio_duration = audio_tensor.shape[-1] / 24000
        
        # 3. 임시 파일로 저장 (Whisper 처리용)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            ta.save(tmp_path, audio_tensor, 24000)
        
        try:
            # 4. Whisper로 초기 타임스탬프 추출
            whisper_timestamps = self._extract_whisper_timestamps(
                tmp_path,
                processed_text,
                original_words
            )
            
            # 5. 원본 텍스트와 정렬 및 연속 타임스탬프 생성
            continuous_timestamps = self._create_continuous_timestamps(
                original_words,
                whisper_timestamps,
                audio_duration
            )
            
        finally:
            # 임시 파일 삭제
            Path(tmp_path).unlink(missing_ok=True)
        
        logger.info(f"연속 타임스탬프 {len(continuous_timestamps)}개 생성 (단어 수: {len(original_words)})")

        return audio_tensor, continuous_timestamps
    
    def _extract_whisper_timestamps(
        self,
        audio_path: str,
        processed_text: str,
    ) -> List[Dict]:
        """Whisper를 사용한 초기 타임스탬프 추출"""
        
        logger.info("Whisper로 타임스탬프를 추출합니다...")
        
        # Whisper 전사 (initial_prompt로 강제 유도)
        result = self.whisper_model.transcribe(
            audio_path,
            language="ko",
            word_timestamps=True,
            initial_prompt=processed_text,  # 원본 텍스트로 유도
            temperature=0.0,  # 결정적 출력
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            condition_on_previous_text=True,
            suppress_tokens=[-1]  # 불필요한 토큰 억제
        )
        
        # Whisper 단어 추출
        whisper_words = []
        for segment in result.get("segments", []):
            whisper_words.extend(segment.get("words", []))
        
        return whisper_words
    
    def _create_continuous_timestamps(
        self,
        original_words: List[str],
        whisper_words: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """원본 텍스트와 정렬하여 연속된 타임스탬프 생성"""
        
        # 1. 원본 단어와 Whisper 단어 매칭
        matched_timestamps = self._match_words(original_words, whisper_words, audio_duration)
        
        # 2. 누락된 단어 처리 및 시간 보간
        all_timestamps = self._interpolate_missing_words(
            original_words, 
            matched_timestamps, 
            audio_duration
        )
        
        # 3. 연속된 타임스탬프로 변환
        continuous_timestamps = []
        
        for i, word in enumerate(original_words):
            # 현재 단어의 시작 시간
            if i == 0:
                start_time = 0.0
            else:
                # 이전 단어의 끝 시간이 현재 단어의 시작 시간
                start_time = continuous_timestamps[i-1]["end"]
            
            # 현재 단어의 끝 시간
            if i < len(original_words) - 1:
                # 다음 단어가 있는 경우: 다음 단어의 시작 시간
                if i + 1 < len(all_timestamps):
                    # 원래 Whisper가 감지한 다음 단어 시작 시간 사용
                    next_start = all_timestamps[i + 1].get("original_start", 
                                                           all_timestamps[i + 1]["start"])
                    end_time = next_start
                else:
                    # 평균 단어 길이로 추정
                    avg_duration = audio_duration / len(original_words)
                    end_time = start_time + avg_duration
            else:
                # 마지막 단어: 오디오 끝까지
                end_time = audio_duration
            
            # 시간 유효성 검증
            if end_time <= start_time:
                end_time = start_time + 0.1  # 최소 0.1초
            
            if end_time > audio_duration:
                end_time = audio_duration
            
            continuous_timestamps.append({
                "word": word,
                "start": round(start_time, 3),
                "end": round(end_time, 3)
            })
        
        return continuous_timestamps
    
    def _match_words(
        self, 
        original_words: List[str], 
        whisper_words: List[Dict]
        ,
        audio_duration: float
    ) -> List[Dict]:
        """DTW를 사용한 원본 단어와 Whisper 단어 매칭"""
        
        if not whisper_words:
            # Whisper가 아무것도 감지하지 못한 경우: 오디오 전체 길이를 사용해 균등 분할
            return self._create_uniform_timestamps(original_words, audio_duration)
        
        matched = []
        whisper_idx = 0
        
        for orig_word in original_words:
            best_match = None
            best_score = 0.0
            best_idx = -1
            
            # 현재 위치 근처에서 최적 매칭 검색
            search_range = min(5, len(whisper_words) - whisper_idx)
            
            for i in range(whisper_idx, min(whisper_idx + search_range, len(whisper_words))):
                whisper_word = whisper_words[i].get("word", "").strip()
                similarity = self._calculate_similarity(orig_word, whisper_word)
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = whisper_words[i]
                    best_idx = i
            
            if best_match and best_score > 0.6:  # 60% 이상 유사도
                matched.append({
                    "word": orig_word,
                    "start": best_match.get("start", 0.0),
                    "original_start": best_match.get("start", 0.0),  # 원래 시작 시간 보존
                    "end": best_match.get("end", 0.0),
                    "confidence": best_score
                })
                whisper_idx = best_idx + 1
            else:
                # 매칭 실패 - 나중에 보간
                matched.append({
                    "word": orig_word,
                    "start": None,
                    "end": None,
                    "confidence": 0.0
                })
        
        return matched
    
    def _interpolate_missing_words(
        self,
        original_words: List[str],
        matched_timestamps: List[Dict],
        audio_duration: float
    ) -> List[Dict]:
        """누락된 단어의 타임스탬프 보간"""
        
        # None 값을 가진 타임스탬프 보간 (안전한 수치 처리)
        for i, ts in enumerate(matched_timestamps):
            if ts["start"] is None:
                # 이전과 다음 유효한 타임스탬프와 인덱스 찾기
                prev_idx = None
                next_idx = None

                for j in range(i - 1, -1, -1):
                    if matched_timestamps[j]["start"] is not None:
                        prev_idx = j
                        break

                for j in range(i + 1, len(matched_timestamps)):
                    if matched_timestamps[j]["start"] is not None:
                        next_idx = j
                        break

                prev_valid = matched_timestamps[prev_idx] if prev_idx is not None else None
                next_valid = matched_timestamps[next_idx] if next_idx is not None else None

                # 보간
                if prev_valid and next_valid:
                    # 이전 유효 종료 시간 확보 (end가 None이면 start 또는 original_start로 대체)
                    prev_end = prev_valid.get("end")
                    if prev_end is None:
                        prev_end = prev_valid.get("original_start", prev_valid.get("start", 0.0))

                    next_start = next_valid.get("start", prev_end)

                    words_between = next_idx - prev_idx - 1
                    if words_between > 0:
                        gap = next_start - prev_end
                        # 안전장치: gap이 음수이면 아주 작은 양수로 대체
                        if gap <= 0:
                            gap = 0.001 * words_between
                        word_duration = gap / words_between
                        position = i - prev_idx
                        ts["start"] = prev_end + (position - 1) * word_duration
                        ts["original_start"] = ts["start"]
                    else:
                        ts["start"] = prev_end
                        ts["original_start"] = ts["start"]

                elif prev_valid:
                    # 이전 값 기준
                    prev_end = prev_valid.get("end") if prev_valid.get("end") is not None else prev_valid.get("original_start", prev_valid.get("start", 0.0))
                    ts["start"] = prev_end
                    ts["original_start"] = ts["start"]

                elif next_valid:
                    # 다음 값 기준
                    avg_duration = audio_duration / max(1, len(original_words))
                    next_start = next_valid.get("start", 0.0)
                    ts["start"] = max(0.0, next_start - avg_duration)
                    ts["original_start"] = ts["start"]

                else:
                    # 기본값 (균등 분할)
                    ts["start"] = i * (audio_duration / max(1, len(original_words)))
                    ts["original_start"] = ts["start"]

                ts["confidence"] = 0.3  # 보간된 값
        
        return matched_timestamps
    
    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """두 단어의 유사도 계산"""
        
        # 정확히 일치
        if word1 == word2:
            return 1.0
        
        # 공백 제거 후 일치
        if word1.replace(" ", "") == word2.replace(" ", ""):
            return 0.95
        
        # 편집 거리 기반 유사도
        return SequenceMatcher(None, word1, word2).ratio()
    
    def _create_uniform_timestamps(
        self, 
        words: List[str], 
        total_duration: float
    ) -> List[Dict]:
        """균등한 타임스탬프 생성 (폴백용)"""
        
        word_duration = total_duration / len(words)
        timestamps = []
        
        for i, word in enumerate(words):
            timestamps.append({
                "word": word,
                "start": i * word_duration,
                "original_start": i * word_duration,
                "end": (i + 1) * word_duration,
                "confidence": 0.1
            })
        
        return timestamps


# 사용 예시
if __name__ == "__main__":
    # 초기화 (기본 Whisper 모델: small)
    tts_system = TTSTimestamp(whisper_model="medium")

    # 텍스트
    text = "안녕하세요 여러분 오늘은 정말 좋은 날입니다"

    # 음성 및 타임스탬프 생성
    audio_tensor, timestamps = tts_system.generate(text, language_id="ko")

    # 결과 로그
    logger.info("\n=== 연속 타임스탬프 ===")
    for i, ts in enumerate(timestamps):
        logger.info(f"{i+1}. {ts['word']}: {ts['start']:.3f}s - {ts['end']:.3f}s")

        # 연속성 검증
        if i > 0:
            gap = timestamps[i]['start'] - timestamps[i-1]['end']
            if abs(gap) > 0.001:
                logger.warning(f"간격 탐지: {gap:.3f}s")

    # 전체 오디오 길이와 마지막 타임스탬프 확인 (샘플레이트 24000)
    audio_duration = audio_tensor.shape[-1] / 24000
    logger.info(f"\n오디오 길이: {audio_duration:.3f}s")
    if timestamps:
        logger.info(f"마지막 타임스탬프 종료: {timestamps[-1]['end']:.3f}s")
