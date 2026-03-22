"""
utils.py — Shared helper functions
"""

import os
import shutil
from pathlib import Path


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def check_file_exists(path: str) -> bool:
    """Return True if the file exists and is not empty."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def clear_dir(path: str, keep_dir: bool = True) -> None:
    """Delete all files inside a directory. Optionally keep the dir itself."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    if keep_dir:
        os.makedirs(path, exist_ok=True)


def list_wav_files(folder: str) -> list[str]:
    """Return sorted list of .wav paths in a folder."""
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ])


def format_duration(seconds: float) -> str:
    """Format seconds as mm:ss string."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def audio_duration(path: str) -> float:
    """Return duration of a WAV file in seconds without loading it fully."""
    import soundfile as sf
    info = sf.info(path)
    return info.duration