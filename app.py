"""
app.py — Streamlit UI for the Voice Cloning System

Run with: streamlit run app.py

Features:
  - Upload audio file OR record from microphone
  - Preprocessing + embedding extraction with progress display
  - Text input with emotion/style selection
  - Language selection (17 languages via XTTS-v2)
  - Non-verbal cue helper (inserts <breath>, <pause> markers)
  - Waveform visualization
  - Voice consistency score display
  - Playback + download of generated audio
  - Reset button to clear session state
"""

import os
import time
import tempfile
import numpy as np
import soundfile as sf
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Voice Clone Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Cards */
.vc-card {
    background: #1a1d27;
    border: 1px solid #2d3147;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Step headers */
.step-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}
.step-num {
    background: #5c6bc0;
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 600;
    flex-shrink: 0;
}
.step-title {
    font-size: 16px;
    font-weight: 600;
    color: #e0e0e0;
}

/* Score badge */
.score-badge {
    display: inline-block;
    background: #1e3a2f;
    color: #4caf85;
    border: 1px solid #2d6b50;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
}

/* Emotion pill buttons */
div[data-testid="column"] button {
    border-radius: 20px !important;
    font-size: 13px !important;
}

/* Hide default Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────
def init_state():
    defaults = {
        "reference_wav": None,       # path to the chosen reference clip
        "embedding_done": False,
        "consistency_score": None,
        "output_path": None,
        "encoder": None,
        "tts": None,
        "vocoder": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Lazy model loading (cached so models load once) ─────────────────────────
@st.cache_resource(show_spinner=False)
def load_encoder():
    from services.ecapa_service import WavLMSpeakerEncoder
    return WavLMSpeakerEncoder()

@st.cache_resource(show_spinner=False)
def load_tts():
    from services.tts_service import TTSService
    return TTSService()

@st.cache_resource(show_spinner=False)
def load_vocoder():
    from services.vocoder_service import VocoderService
    return VocoderService()


# ── Helpers ──────────────────────────────────────────────────────────────────
def draw_waveform(audio_path: str):
    """Draw a simple waveform using st.line_chart."""
    import librosa
    audio, sr = librosa.load(audio_path, sr=None, mono=True, duration=30)
    # Downsample for display
    display_audio = audio[::max(1, len(audio) // 1000)]
    st.line_chart(display_audio, height=80, use_container_width=True)


def save_uploaded_audio(uploaded_file) -> str:
    """Save an uploaded file to a temp location and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(uploaded_file.read())
        return f.name


def reset_session():
    """Clear all session state and trigger a rerun."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ Voice Clone Studio")
    st.caption("Zero-shot voice cloning powered by XTTS-v2")
    st.divider()

    st.markdown("### Settings")
    language = st.selectbox(
        "Language",
        options=[
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
        ],
        index=0,
        help="Language for the synthesized output",
    )

    style = st.selectbox(
        "Voice style",
        options=["neutral", "calm", "excited", "sad", "whisper", "fast", "slow"],
        index=0,
    )

    post_deess = st.toggle("De-esser", value=True, help="Reduce harsh sibilance (ssss sounds)")
    post_breaths = st.toggle("Add breath sounds", value=False, help="Insert subtle breaths at pauses")

    st.divider()

    st.markdown("### Non-verbal cues")
    st.caption("Add these markers anywhere in your text:")
    st.code("<breath>   — inhale\n<pause>    — longer pause\n<hmm>      — filler\n<sigh>     — sigh")

    st.divider()
    if st.button("🔄 Reset everything", use_container_width=True):
        reset_session()


# ── Main content ─────────────────────────────────────────────────────────────
st.markdown("# 🎙️ Voice Clone Studio")
st.markdown("Clone any voice in seconds. Upload a sample, type your text, generate.")
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

# ════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Input
# ════════════════════════════════════════════════════════════════════════
with col_left:

    # ── Step 1: Voice sample ────────────────────────────────────────────
    st.markdown("""
    <div class="vc-card">
    <div class="step-header">
        <div class="step-num">1</div>
        <div class="step-title">Provide your voice sample</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio(
        "Input method",
        ["Upload audio file", "Record from microphone"],
        horizontal=True,
        label_visibility="collapsed",
    )

    raw_audio_path = None

    if input_method == "Upload audio file":
        uploaded = st.file_uploader(
            "Upload WAV, MP3, FLAC, or M4A",
            type=["wav", "mp3", "flac", "m4a", "ogg"],
            label_visibility="collapsed",
        )
        if uploaded:
            raw_audio_path = save_uploaded_audio(uploaded)
            st.success(f"Uploaded: {uploaded.name}")

    else:
        st.info("🎤 Click below to record. Aim for 30–60 seconds of clear speech.")
        recorded = st.audio_input("Record your voice", label_visibility="collapsed")
        if recorded:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(recorded.read())
                raw_audio_path = f.name
            st.success("Recording captured!")

    # Show waveform + preprocess button if we have audio
    if raw_audio_path:
        st.markdown("**Preview:**")
        st.audio(raw_audio_path)

        if st.button("⚙️ Preprocess & Extract Voice", type="primary", use_container_width=True):
            with st.status("Processing your voice sample...", expanded=True) as status:

                st.write("🔊 Loading audio...")
                from scripts.preprocess import preprocess_audio

                st.write("🧹 Denoising + VAD...")
                preprocessed_dir = "data/preprocessed_audio"
                chunks = preprocess_audio(raw_audio_path, preprocessed_dir, denoise=True)

                if not chunks:
                    st.error("No speech detected. Try a longer or louder recording.")
                    st.stop()

                st.write(f"✂️ Created {len(chunks)} audio chunks")
                st.write("🧠 Extracting speaker embedding (WavLM-Large)...")

                encoder = load_encoder()
                emb_path = "data/embeddings/speaker_avg.npy"
                os.makedirs("data/embeddings", exist_ok=True)
                avg_emb = encoder.encode_folder(preprocessed_dir, save_path=emb_path)

                # Pick the best chunk as the TTS reference (closest to average)
                best_chunk = chunks[0]
                best_sim = -1
                for c in chunks:
                    emb = encoder.encode(c)
                    sim = encoder.speaker_similarity(emb, avg_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_chunk = c

                st.session_state.reference_wav = best_chunk
                st.session_state.consistency_score = best_sim
                st.session_state.embedding_done = True

                status.update(label="Voice extracted!", state="complete")

    # Show consistency score
    if st.session_state.embedding_done:
        score = st.session_state.consistency_score or 0
        score_pct = int(score * 100)
        color = "#4caf85" if score > 0.85 else "#f5a623" if score > 0.7 else "#e05252"
        st.markdown(
            f'<div class="score-badge" style="border-color:{color};color:{color}">'
            f'Voice consistency: {score_pct}%</div>',
            unsafe_allow_html=True,
        )
        if score < 0.7:
            st.warning("Low consistency — try recording in a quieter environment with longer speech.")


# ════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Text + Generate
# ════════════════════════════════════════════════════════════════════════
with col_right:

    # ── Step 2: Text input ──────────────────────────────────────────────
    st.markdown("""
    <div class="vc-card">
    <div class="step-header">
        <div class="step-num">2</div>
        <div class="step-title">Type the text to synthesize</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        "Your text",
        value="Hello! This is my cloned voice speaking. <breath> Pretty amazing, right?",
        height=160,
        label_visibility="collapsed",
        help="Use <breath>, <pause>, <hmm>, <sigh> markers for natural speech.",
        placeholder="Type anything here. Use <breath>, <pause>, <hmm> for natural cues.",
    )

    char_count = len(text_input)
    st.caption(f"{char_count} characters")

    # Quick cue insert buttons
    st.caption("Insert cue:")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("💨 Breath"): text_input += " <breath>"
    with b2:
        if st.button("⏸️ Pause"): text_input += " <pause>"
    with b3:
        if st.button("🤔 Hmm"): text_input += " <hmm>"
    with b4:
        if st.button("😮‍💨 Sigh"): text_input += " <sigh>"

    st.divider()

    # ── Step 3: Generate ────────────────────────────────────────────────
    st.markdown("""
    <div class="vc-card">
    <div class="step-header">
        <div class="step-num">3</div>
        <div class="step-title">Generate your cloned voice</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    can_generate = st.session_state.embedding_done and text_input.strip()

    if not st.session_state.embedding_done:
        st.info("Complete Step 1 first to extract your voice.")

    generate_btn = st.button(
        "🚀 Generate Speech",
        type="primary",
        use_container_width=True,
        disabled=not can_generate,
    )

    if generate_btn and can_generate:
        with st.status("Synthesizing speech...", expanded=True) as status:
            tts = load_tts()
            vocoder = load_vocoder()

            raw_output = "data/outputs/raw_clone.wav"
            final_output = "data/outputs/final_clone.wav"
            os.makedirs("data/outputs", exist_ok=True)

            st.write(f"🎤 Synthesizing in language: {language}, style: {style}...")
            t0 = time.time()

            tts.synthesize(
                text=text_input,
                reference_wav=st.session_state.reference_wav,
                output_path=raw_output,
                language=language,
                style=style,
            )

            st.write("🎛️ Applying post-processing...")
            vocoder.process(
                input_path=raw_output,
                output_path=final_output,
                deess=post_deess,
                add_breaths=post_breaths,
            )

            elapsed = time.time() - t0
            st.session_state.output_path = final_output

            status.update(
                label=f"Done! Generated in {elapsed:.1f}s",
                state="complete",
            )

    # ── Output player ────────────────────────────────────────────────────
    if st.session_state.output_path and os.path.isfile(st.session_state.output_path):
        st.markdown("### 🔊 Output")
        st.audio(st.session_state.output_path)

        st.markdown("**Waveform:**")
        draw_waveform(st.session_state.output_path)

        with open(st.session_state.output_path, "rb") as f:
            st.download_button(
                "⬇️ Download WAV",
                data=f,
                file_name="cloned_voice.wav",
                mime="audio/wav",
                use_container_width=True,
            )


# ── Bottom info bar ──────────────────────────────────────────────────────────
st.divider()
info_cols = st.columns(4)
with info_cols[0]:
    st.metric("TTS Engine", "XTTS-v2")
with info_cols[1]:
    st.metric("Speaker Encoder", "WavLM-Large")
with info_cols[2]:
    st.metric("Sample Rate", "24 kHz")
with info_cols[3]:
    st.metric("Languages", "17")