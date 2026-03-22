"""
ecapa_service.py — Upgraded to WavLM-Large speaker embeddings

Changes from original:
  - WavLM-Large replaces ECAPA-TDNN (much richer speaker representation)
  - Keeps ECAPA as fallback if WavLM is unavailable
  - Voice consistency scoring between reference and generated audio
  - Cosine similarity check on averaged embeddings
"""

import os
import torch
import numpy as np
import librosa
from typing import Optional


class WavLMSpeakerEncoder:
    """
    Speaker encoder using Microsoft WavLM-Large.
    Produces 1024-dim embeddings — richer than ECAPA's 192-dim.
    Mean-pools the transformer output across time for a fixed-size vector.
    """

    MODEL_ID = "microsoft/wavlm-large"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        try:
            from transformers import WavLMModel, AutoFeatureExtractor
            print(f"[encoder] Loading WavLM-Large on {self.device}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
            self.model = WavLMModel.from_pretrained(self.MODEL_ID).to(self.device)
            self.model.eval()
            self.backend = "wavlm"
            print("[encoder] WavLM-Large loaded.")
        except Exception as e:
            print(f"[encoder] WavLM failed ({e}), falling back to ECAPA-TDNN.")
            self._load_ecapa_fallback()

    def _load_ecapa_fallback(self):
        from speechbrain.inference.speaker import EncoderClassifier
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )
        self.backend = "ecapa"
        print("[encoder] ECAPA-TDNN loaded as fallback.")

    def encode(self, audio_path: str) -> np.ndarray:
        """Extract speaker embedding from a single audio file."""
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        if self.backend == "wavlm":
            return self._encode_wavlm(audio)
        else:
            return self._encode_ecapa(audio_path)

    def _encode_wavlm(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            # Mean-pool across time dimension → (1, hidden_size)
            embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.squeeze().cpu().numpy()

    def _encode_ecapa(self, audio_path: str) -> np.ndarray:
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        waveform = torch.tensor(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform)
        return embedding.squeeze().cpu().numpy()

    def encode_folder(self, folder_path: str, save_path: Optional[str] = None) -> np.ndarray:
        """
        Extract and average embeddings for all chunks in a folder.
        Optionally saves the averaged embedding as a .npy file.
        """
        embeddings = []

        wav_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith(".wav") and f.startswith("chunk_")
        ])

        if not wav_files:
            raise ValueError(f"No audio chunks found in: {folder_path}")

        for fname in wav_files:
            path = os.path.join(folder_path, fname)
            try:
                emb = self.encode(path)
                embeddings.append(emb)
            except Exception as e:
                print(f"[encoder] Skipping {fname}: {e}")

        if not embeddings:
            raise ValueError("All chunks failed encoding.")

        avg_embedding = np.mean(embeddings, axis=0)

        # Voice consistency score — std dev of cosine sims to the mean
        consistency = self._consistency_score(embeddings, avg_embedding)
        print(f"[encoder] Voice consistency score: {consistency:.3f} (higher = more consistent)")

        if save_path:
            np.save(save_path, avg_embedding)
            print(f"[encoder] Embedding saved to: {save_path}")

        return avg_embedding

    def _consistency_score(self, embeddings: list, avg: np.ndarray) -> float:
        """
        Cosine similarity of each chunk embedding to the average.
        Score close to 1.0 = very consistent speaker across chunks.
        """
        sims = []
        avg_norm = avg / (np.linalg.norm(avg) + 1e-8)
        for emb in embeddings:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sims.append(float(np.dot(emb_norm, avg_norm)))
        return float(np.mean(sims))

    def speaker_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two speaker embeddings. Range: -1 to 1."""
        a = emb_a / (np.linalg.norm(emb_a) + 1e-8)
        b = emb_b / (np.linalg.norm(emb_b) + 1e-8)
        return float(np.dot(a, b))


# Convenience alias for drop-in compatibility with original code
ECAPASpeakerEncoder = WavLMSpeakerEncoder