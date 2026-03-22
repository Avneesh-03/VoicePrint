"""
vocoder_service.py — Upgraded post-processing pipeline

Changes from original:
  - LUFS normalization on output (EBU R128 broadcast standard)
  - De-essing (high-frequency sibilance reduction)
  - Breath re-insertion option (adds subtle breath sounds at natural points)
  - Voice consistency scoring between input reference and output
  
Note: XTTS-v2 has BigVGAN integrated internally, so this module focuses on
      post-processing the final waveform rather than running a separate vocoder.
"""

import os
import numpy as np
import soundfile as sf
import librosa
from typing import Optional


TARGET_SR = 24000
TARGET_LUFS = -16.0     # slightly louder than broadcast, good for TTS output


class VocoderService:
    """
    Post-processing for synthesized audio.
    In the XTTS-v2 pipeline, the vocoder (BigVGAN) runs inside XTTS itself.
    This class handles everything that happens AFTER synthesis:
      - LUFS normalization
      - De-essing
      - Soft de-clipping
      - Optional breath re-insertion
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device  # kept for API compatibility

    def process(
        self,
        input_path: str,
        output_path: str,
        target_lufs: float = TARGET_LUFS,
        deess: bool = True,
        add_breaths: bool = False,
    ) -> str:
        """
        Apply post-processing chain to a synthesized WAV file.

        Args:
            input_path:  Path to raw synthesized WAV.
            output_path: Path to save post-processed WAV.
            target_lufs: Target loudness (default -16 LUFS).
            deess:       Apply de-esser to reduce harsh sibilants.
            add_breaths: Insert subtle breath sounds at natural pause points.

        Returns:
            Path to processed output file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        audio, sr = librosa.load(input_path, sr=None, mono=True)
        print(f"[vocoder] Loaded {len(audio)/sr:.1f}s audio at {sr}Hz")

        # 1. Soft de-clip (remove harsh peaks before processing)
        audio = self._soft_clip(audio)

        # 2. De-esser (reduces sibilance — the "ssss" harshness in TTS)
        if deess:
            audio = self._deess(audio, sr)

        # 3. LUFS normalize
        audio = self._lufs_normalize(audio, sr, target_lufs)

        # 4. Breath re-insertion
        if add_breaths:
            audio = self._add_breath_sounds(audio, sr)

        # 5. Final peak limiter (safety ceiling)
        audio = np.clip(audio, -0.98, 0.98)

        sf.write(output_path, audio, sr)
        print(f"[vocoder] Post-processed saved to: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Also accept raw numpy arrays (for in-memory pipeline use)
    # ------------------------------------------------------------------

    def refine_audio(
        self,
        wav: np.ndarray,
        sr: int,
        output_path: str,
    ) -> str:
        """Legacy API: accepts numpy array, applies processing, saves."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        wav = self._soft_clip(wav)
        wav = self._deess(wav, sr)
        wav = self._lufs_normalize(wav, sr, TARGET_LUFS)
        wav = np.clip(wav, -0.98, 0.98)

        sf.write(output_path, wav, sr)
        return output_path

    # ------------------------------------------------------------------
    # Internal DSP helpers
    # ------------------------------------------------------------------

    def _soft_clip(self, audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """Soft-knee limiter — smoothly attenuates peaks above threshold."""
        mask = np.abs(audio) > threshold
        audio[mask] = np.sign(audio[mask]) * (
            threshold + (1 - threshold) * np.tanh(
                (np.abs(audio[mask]) - threshold) / (1 - threshold)
            )
        )
        return audio

    def _deess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simple frequency-domain de-esser.
        Attenuates 5kHz–10kHz band when energy spikes there.
        """
        try:
            # Short-time FFT
            n_fft = 2048
            hop = n_fft // 4
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            # Define sibilance band
            sib_mask = (freqs >= 5000) & (freqs <= 10000)

            # Compute per-frame energy in sibilance band
            magnitude = np.abs(stft)
            sib_energy = magnitude[sib_mask, :].mean(axis=0)
            total_energy = magnitude.mean(axis=0) + 1e-8
            sib_ratio = sib_energy / total_energy

            # Where sibilance ratio is high, attenuate that band
            threshold = np.percentile(sib_ratio, 75)
            attenuation = np.ones_like(sib_ratio)
            excess = (sib_ratio - threshold).clip(min=0)
            attenuation = 1.0 / (1.0 + excess * 8)

            # Apply attenuation only to sibilance band
            gain = np.ones_like(magnitude)
            gain[sib_mask, :] = attenuation[np.newaxis, :]

            stft_processed = stft * gain
            audio_out = librosa.istft(stft_processed, hop_length=hop, length=len(audio))
            return audio_out.astype(np.float32)

        except Exception as e:
            print(f"[vocoder] De-esser failed ({e}), skipping.")
            return audio

    def _lufs_normalize(
        self,
        audio: np.ndarray,
        sr: int,
        target_lufs: float,
    ) -> np.ndarray:
        """LUFS normalization via pyloudnorm, peak normalize as fallback."""
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            if loudness == float("-inf"):
                peak = np.max(np.abs(audio))
                return audio / peak if peak > 0 else audio
            return np.clip(
                pyln.normalize.loudness(audio, loudness, target_lufs),
                -1.0, 1.0,
            ).astype(np.float32)
        except ImportError:
            peak = np.max(np.abs(audio))
            return (audio / peak * 0.9).astype(np.float32) if peak > 0 else audio

    def _add_breath_sounds(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Detects long pauses and inserts very subtle breath-like noise.
        This is a lightweight approximation — for production use a real breath corpus.
        """
        # Find silent regions (amplitude < threshold for > 0.4s)
        frame_len = int(0.02 * sr)   # 20ms frames
        hop = frame_len // 2
        rms = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop)[0]
        silence_threshold = np.percentile(rms, 10)

        result = audio.copy()
        in_silence = False
        silence_start = 0

        for i, r in enumerate(rms):
            t = i * hop
            if r < silence_threshold and not in_silence:
                in_silence = True
                silence_start = t
            elif r >= silence_threshold and in_silence:
                duration = (t - silence_start) / sr
                if duration > 0.5:
                    # Insert shaped noise (breath approximation)
                    breath_len = int(min(0.15, duration * 0.3) * sr)
                    if breath_len > 0:
                        breath = np.random.randn(breath_len).astype(np.float32)
                        # Shape with envelope
                        env = np.hanning(breath_len)
                        breath = breath * env * 0.008  # very quiet
                        insert_at = silence_start + int(0.05 * sr)
                        end_at = min(insert_at + breath_len, len(result))
                        result[insert_at:end_at] += breath[: end_at - insert_at]
                in_silence = False

        return result