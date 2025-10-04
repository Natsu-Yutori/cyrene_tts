import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class PostProcessConfig:
    sample_rate: int
    window_ms: float = 30.0
    hop_ms: float = 10.0
    lead_buffer_ms: float = 30.0
    tail_buffer_ms: float = 160.0
    fade_out_ms: float = 60.0
    min_duration_ms: float = 250.0
    noise_percentile: float = 0.2
    dynamic_ratio: float = 0.2
    min_threshold: float = 1e-3
    silence_hangover_ms: float = 120.0


class AudioPostProcessor:
    """Adaptive audio post-processing utilities for trimming trailing silence and noise."""

    def __init__(self, sample_rate: int, config: Optional[PostProcessConfig] = None):
        if config is None:
            config = PostProcessConfig(sample_rate=sample_rate)
        elif config.sample_rate != sample_rate:
            config = dataclasses.replace(config, sample_rate=sample_rate)
        self.cfg = config

    def apply(self, waveform: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Trim trailing silence/noise and apply a gentle fade-out to avoid abrupt endings.

        Args:
            waveform: Tensor shaped as [channels, time] or [time].
            sample_rate: Optional override of the sample rate.

        Returns:
            Processed waveform tensor on CPU.
        """
        if waveform is None:
            return waveform

        sr = sample_rate or self.cfg.sample_rate
        if sr <= 0:
            raise ValueError("Invalid sample rate for post-processing")

        original_dtype = waveform.dtype
        added_channel_dim = False

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            added_channel_dim = True

        waveform_cpu = waveform.detach().to("cpu")
        trimmed = self._trim_silence(waveform_cpu, sr)

        if trimmed.shape[-1] == 0:
            # Fallback to original if trimming removed everything
            trimmed = waveform_cpu

        faded = self._apply_fade_out(trimmed, sr)

        if faded.shape[-1] == 0:
            faded = trimmed

        if added_channel_dim:
            faded = faded.squeeze(0)

        return faded.to(dtype=original_dtype)

    def _trim_silence(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        cfg = self.cfg
        window_samples = max(int(sample_rate * (cfg.window_ms / 1000.0)), 1)
        hop_samples = max(int(sample_rate * (cfg.hop_ms / 1000.0)), 1)
        lead_buffer = int(sample_rate * (cfg.lead_buffer_ms / 1000.0))
        tail_buffer = int(sample_rate * (cfg.tail_buffer_ms / 1000.0))
        min_samples = int(sample_rate * (cfg.min_duration_ms / 1000.0))

        mono = waveform.mean(dim=0, keepdim=True)
        padded = F.pad(mono, (0, max(0, window_samples - 1)))
        energy = F.avg_pool1d(
            padded.pow(2).unsqueeze(0),
            kernel_size=window_samples,
            stride=hop_samples,
            ceil_mode=True,
        ).squeeze(0).squeeze(0)

        if energy.numel() == 0:
            return waveform

        rms = torch.sqrt(energy + 1e-8)
        sorted_rms, _ = torch.sort(rms)
        percentile_index = max(int(sorted_rms.numel() * cfg.noise_percentile) - 1, 0)
        noise_floor = sorted_rms[percentile_index]
        max_rms = torch.max(rms)
        dynamic_threshold = noise_floor + (max_rms - noise_floor) * cfg.dynamic_ratio
        dynamic_threshold = torch.clamp(dynamic_threshold, min=cfg.min_threshold)

        raw_active_frames = (rms > dynamic_threshold).cpu().tolist()
        total_samples = waveform.shape[-1]
        silence_frames_required = max(int(round(cfg.silence_hangover_ms / cfg.hop_ms)), 1)

        if not any(raw_active_frames):
            keep_samples = min(total_samples, max(min_samples, window_samples))
            return waveform[..., :keep_samples]

        active_frames = self._smooth_activity(raw_active_frames, silence_frames_required)

        first_active_idx = active_frames.index(True)
        last_active_idx = len(active_frames) - 1 - active_frames[::-1].index(True)

        leading_silence_frames = first_active_idx
        trailing_silence_frames = len(active_frames) - 1 - last_active_idx

        start_frame = max(0, leading_silence_frames - silence_frames_required)
        end_frame = len(active_frames) - max(0, trailing_silence_frames - silence_frames_required)
        end_frame = max(end_frame, start_frame + 1)

        start_sample = max(0, start_frame * hop_samples - lead_buffer)
        end_sample = min(total_samples, end_frame * hop_samples + window_samples + tail_buffer)

        if end_sample <= start_sample:
            return waveform[..., :max(min_samples, window_samples)]

        segment_samples = end_sample - start_sample
        if segment_samples < min_samples:
            deficit = min_samples - segment_samples
            pad_front = deficit // 2
            pad_back = deficit - pad_front
            start_sample = max(0, start_sample - pad_front)
            end_sample = min(total_samples, end_sample + pad_back)

        return waveform[..., start_sample:end_sample]

    @staticmethod
    def _smooth_activity(activity: list[bool], hangover: int) -> list[bool]:
        if hangover <= 0 or len(activity) == 0:
            return activity

        forward: list[bool] = []
        hold = 0
        for is_active in activity:
            if is_active:
                hold = hangover
            else:
                hold = max(hold - 1, 0)
            forward.append(is_active or hold > 0)

        backward = [False] * len(activity)
        hold = 0
        for idx in range(len(activity) - 1, -1, -1):
            is_active = activity[idx]
            if is_active:
                hold = hangover
            else:
                hold = max(hold - 1, 0)
            backward[idx] = is_active or hold > 0

        return [f or b for f, b in zip(forward, backward)]

    def _apply_fade_out(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        fade_samples = int(sample_rate * (self.cfg.fade_out_ms / 1000.0))
        if fade_samples <= 0 or waveform.shape[-1] <= fade_samples:
            return waveform

        fade_curve = torch.linspace(1.0, 0.0, fade_samples, dtype=waveform.dtype)
        waveform = waveform.clone()
        waveform[..., -fade_samples:] *= fade_curve
        return waveform
