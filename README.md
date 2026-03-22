# 🎙️ Project VoicePrint
> *So deliciously cloned, even close ones can't figure it out.*

![Python](https://img.shields.io/badge/Python-3.12-blue) 
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange) 
![License](https://img.shields.io/badge/License-Open%20Source-green)

## 📌 What is it?
A zero-shot AI voice cloning system that replicates any speaker's voice from just 3–30 seconds of audio — no fine-tuning, no large datasets required. Built on Google Colab with a Streamlit web interface.

---

## 🧠 Models Used
| Model | By | Role |
|---|---|---|
| XTTSv2 | Coqui AI — Idiap Fork | Zero-shot speech synthesis |
| WavLM Large | Microsoft Research | Speaker embedding (1024-dim) |

---

## ⚡ Key Features
- 🌍 17 languages — Hindi, Arabic, Chinese, Japanese and more
- 🎧 24kHz high-fidelity audio output
- 📊 85%+ voice consistency score
- ⏱️ Works from as little as 3 seconds of reference audio
- 🗣️ Non-verbal cues — `<breath>` `<pause>` `<hmm>` `<sigh>`
- 🎭 Style controls — neutral, calm, excited, sad, whisper, fast, slow
- 📥 WAV download + waveform visualization in browser

---

## 🏗️ System Architecture
```
Audio Input → preprocess.py → ecapa_service.py → tts_service.py → vocoder_service.py → WAV Output
             (Denoise+VAD)    (WavLM 1024-dim)    (XTTSv2 TTS)     (LUFS+De-ess)
```

---

## 📁 Project Structure
```
voice_Cloning/
├── app.py                   # Streamlit web UI — main entry point
├── test.py                  # Standalone test script
├── scripts/
│   ├── preprocess.py        # Denoise, VAD, resample, chunk audio
│   └── utils.py             # Shared helper functions
└── services/
    ├── ecapa_service.py     # WavLM-Large speaker encoder
    ├── tts_service.py       # XTTSv2 synthesis service
    └── vocoder_service.py   # Post-processing pipeline
```

---

## 🚀 How to Run
> Requires Google Colab with T4 GPU (15GB VRAM)

1. Upload project zip to Google Drive
2. Open Google Colab — select **T4 GPU** runtime
3. **Cell 1** — Mount Drive and extract project
4. **Cell 2** — Install dependencies
5. **Cell 3** — Apply XTTSv2 compatibility patch ⚠️ never skip
6. **Cell 4** — Apply TOS auto-accept patch ⚠️ never skip
7. **Cell 5** — Download XTTSv2 model (~1.8GB, first run only)
8. **Cell 6** — Launch Streamlit app via ngrok
9. Open the ngrok URL printed in output

---

## 🛠️ Tech Stack
| Tool          | Version       |          Purpose         |
|---            |---            |---                       |
| coqui-tts     | 0.27.5        | XTTSv2 TTS engine        |
| transformers  | 4.40.0        | WavLM model loading      |
| PyTorch       | 2.10.0+cu128  | Deep learning backend    |
| Streamlit     | 1.35.0        | Web interface            |
| librosa       | 0.11.0        | Audio processing         |
| numpy         | 1.26.4        | Numerical computing      |
| DeepFilterNet | optional      | Noise suppression        |
| Silero VAD    | auto          | Voice activity detection |
| pyngrok       | 7.5.x         | Public URL tunneling     |
| Google Colab  | —             | Cloud GPU runtime        |

---

## ⚠️ Known Issues & Fixes
| Issue | Fix |
|---|---|
| `isin_mps_friendly` ImportError | Patch `autoregressive.py` with `torch.isin()` |
| EOFError TOS freeze | Patch `manage.py` + set `COQUI_TOS_AGREED=1` |
| VRAM exceeded on laptop | Use Google Colab T4 GPU |
| numpy 2.x breaks librosa | Pin to `numpy==1.26.4` |
| Robotic voice output | Use clean spoken audio, not singing |

---

## 🎯 Applications
- **Content Creation** — Dubbing, podcasts, audiobooks, YouTube voiceovers
- **Accessibility** — Voice banking for ALS/Parkinson's patients
- **Education** — E-learning narration, AI tutors, language learning
- **Gaming** — NPC voices, interactive fiction, virtual avatars
- **Enterprise** — Customer service bots, corporate communications

---

## 🔮 Future Scope
- Real-time voice conversion from live microphone
- Docker deployment — eliminate Colab dependency
- Singing voice support via RVC/SVC
- Cloud hosting on HuggingFace Spaces or AWS
- Fine-tuned models for Hindi and Indic languages

---

## 👥 Authors
Aviral Tiwari
Avneesh Kumar Mishra
Arushi Agrawal
Yamini Pal

---
*Version 1.0.0 — Built with open-source models only*
