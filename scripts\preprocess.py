"""
preprocess.py — Upgraded preprocessing pipeline

Changes from original:
  - DeepFilterNet for real-time noise suppression (replaces basic trim)
  - Silero VAD for smart voice activity detection (replaces energy-based trim)
  - 24kHz target sample rate (up from 16kHz, matches XTTS-v2 + BigVGAN)
  - LUFS normalization (replaces peak norm — more perceptually consistent)
  - Chunk overlap to avoid cutting words mid-phoneme
"""

import os
import numpy as np
import soundfile as sf
import librosa
import torch

TARGET_SR = 24000          # XTTS-v2 and BigVGAN both prefer 24kHz
CHUNK_DURATION = 8.0       # seconds — sweet spot for XTTS reference clips
OVERLAP_DURATION = 0.5     # seconds of overlap between chunks
MIN_CHUNK_DURATION = 3.0   # skip chunks shorter than this
TARGET_LUFS = -23.0        # EBU R128 broadcast standard


def _load_silero_vad():
    """Load Silero VAD from torch hub (auto-cached after first run)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model, utils


def _denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply DeepFilterNet noise suppression.
    Falls back gracefully if deepfilternet is not installed.
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        import tempfile, soundfile as sf_

        # DeepFilterNet works on files; use a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_in = f.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_out = f.name

        sf_.write(tmp_in, audio, sr)
        model, df_state, _ = init_df()
        audio_df, _ = load_audio(tmp_in, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio_df)
        save_audio(tmp_out, enhanced, df_state.sr())
        result, _ = librosa.load(tmp_out, sr=sr, mono=True)

        os.unlink(tmp_in)
        os.unlink(tmp_out)
        return result

    except ImportError:
        print("[preprocess] DeepFilterNet not installed — skipping denoising.")
        print("  Install with: pip install deepfilternet")
        return audio


def _apply_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Use Silero VAD to keep only speech segments.
    Falls back to librosa energy trim if torch hub fails.
    """
    try:
        model, utils = _load_silero_vad()
        (get_speech_timestamps, _, read_audio, *_) = utils

        # Silero expects 16kHz mono tensor
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        wav_tensor = torch.FloatTensor(audio_16k)

        timestamps = get_speech_timestamps(
            wav_tensor, model,
            sampling_rate=16000,
            threshold=0.4,
            min_silence_duration_ms=300,
            min_speech_duration_ms=250,
        )

        if not timestamps:
            print("[preprocess] VAD found no speech — falling back to trim.")
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
            return audio_trimmed

        # Stitch speech segments, resample back to target SR
        segments = []
        scale = sr / 16000
        for seg in timestamps:
            start = int(seg["start"] * scale)
            end = int(seg["end"] * scale)
            segments.append(audio[start:end])

        return np.concatenate(segments)

    except Exception as e:
        print(f"[preprocess] Silero VAD failed ({e}) — using librosa trim.")
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
        return audio_trimmed


def _lufs_normalize(audio: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> np.ndarray:
    """
    LUFS-based normalization using pyloudnorm.
    Falls back to peak normalization if pyloudnorm is missing.
    """
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if loudness == float("-inf"):
            # Silence or near-silence — just peak normalize
            peak = np.max(np.abs(audio))
            return audio / peak if peak > 0 else audio
        normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
        # Clip to prevent distortion
        return np.clip(normalized, -1.0, 1.0)
    except ImportError:
        print("[preprocess] pyloudnorm not installed — using peak normalization.")
        peak = np.max(np.abs(audio))
        return audio / peak if peak > 0 else audio


def _chunk_audio(audio: np.ndarray, sr: int, output_dir: str) -> list[str]:
    """
    Split audio into overlapping chunks for robust embedding extraction.
    Skips chunks that are too short.
    """
    chunk_samples = int(CHUNK_DURATION * sr)
    overlap_samples = int(OVERLAP_DURATION * sr)
    step = chunk_samples - overlap_samples
    min_samples = int(MIN_CHUNK_DURATION * sr)

    chunks = []
    i = 0
    chunk_idx = 0

    while i < len(audio):
        chunk = audio[i : i + chunk_samples]
        if len(chunk) >= min_samples:
            path = os.path.join(output_dir, f"chunk_{chunk_idx}.wav")
            sf.write(path, chunk, sr)
            chunks.append(path)
            chunk_idx += 1
        i += step

    return chunks


def preprocess_audio(
    input_path: str,
    output_dir: str,
    target_sr: int = TARGET_SR,
    denoise: bool = True,
) -> list[str]:
    """
    Full preprocessing pipeline:
      1. Load + convert to mono
      2. Denoise (DeepFilterNet)
      3. VAD — keep only speech regions (Silero)
      4. Resample to target_sr
      5. LUFS normalize
      6. Chunk with overlap
      7. Save chunks + full processed.wav

    Returns list of chunk file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[preprocess] Loading: {input_path}")
    audio, sr = librosa.load(input_path, sr=None, mono=True)
    print(f"[preprocess] Loaded {len(audio)/sr:.1f}s at {sr}Hz")

    if denoise:
        print("[preprocess] Denoising...")
        audio = _denoise(audio, sr)

    print("[preprocess] Applying VAD...")
    audio = _apply_vad(audio, sr)
    print(f"[preprocess] After VAD: {len(audio)/sr:.1f}s of speech")

    if sr != target_sr:
        print(f"[preprocess] Resampling {sr}Hz → {target_sr}Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    print("[preprocess] Normalizing loudness...")
    audio = _lufs_normalize(audio, target_sr)

    # Save full processed file
    processed_path = os.path.join(output_dir, "processed.wav")
    sf.write(processed_path, audio, target_sr)

    # Chunk it
    print("[preprocess] Chunking audio...")
    chunks = _chunk_audio(audio, target_sr, output_dir)
    print(f"[preprocess] Created {len(chunks)} chunks")

    return chunks