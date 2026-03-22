# quick_test.py — Sequential RAM: encoder first, then TTS
import os, sys, gc, numpy as np, soundfile as sf
import torch

REFERENCE_WAV = "data/preprocessed_audio/chunk_0.wav"  # ← adjust
EMB_NPY       = "data/embeddings/speaker_avg.npy"
OUTPUT_WAV    = "data/outputs/test_output.wav"
TEXT          = "Hello, this is a test of my cloned voice."
LANGUAGE      = "en"

os.makedirs("data/outputs", exist_ok=True)

# ── PHASE 1: Load encoder, extract embedding, then DELETE it ──────────────────
print("[1/3] Loading WavLM encoder...")
from services.ecapa_service import WavLMSpeakerEncoder
encoder = WavLMSpeakerEncoder()

# If embedding already saved, skip re-encoding
if not os.path.exists(EMB_NPY):
    print("[1/3] Encoding speaker from chunks...")
    encoder.encode_folder("data/preprocessed_audio", save_path=EMB_NPY)
else:
    print(f"[1/3] Found existing embedding: {EMB_NPY} — skipping encode")

# ── FREE encoder from RAM completely ─────────────────────────────────────────
print("[1/3] Freeing encoder from RAM...")
del encoder
gc.collect()
torch.cuda.empty_cache()   # no-op on CPU but harmless
print("[1/3] Encoder freed.")

# ── PHASE 2: Load XTTS-v2 (now has more RAM headroom) ────────────────────────
print("[2/3] Loading XTTS-v2... (patience — 2-5 min)")
from services.tts_service import TTSService
tts = TTSService()
tts._load()
print("[2/3] XTTS-v2 ready.")

# ── PHASE 3: Synthesize ───────────────────────────────────────────────────────
print("[3/3] Synthesizing...")
tts.synthesize(
    text=TEXT,
    reference_wav=REFERENCE_WAV,
    output_path=OUTPUT_WAV,
    language=LANGUAGE,
    style="neutral",
)
print(f"[3/3] Done! Audio saved to: {OUTPUT_WAV}")
