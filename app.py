# app.py

import html
import io
import os
import time

import streamlit as st
from azure.cognitiveservices.speech import ResultReason, SpeechConfig, SpeechSynthesizer
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
import os


# =========================================================
# CONFIG
# =========================================================


load_dotenv()

try:
    VISION_KEY = st.secrets["VISION_KEY"]
    VISION_ENDPOINT = st.secrets["VISION_ENDPOINT"]

    SPEECH_KEY = st.secrets["SPEECH_KEY"]
    SPEECH_REGION = st.secrets["SPEECH_REGION"]

except Exception:
    VISION_KEY = os.getenv("VISION_KEY")
    VISION_ENDPOINT = os.getenv("VISION_ENDPOINT")

    SPEECH_KEY = os.getenv("SPEECH_KEY")
    SPEECH_REGION = os.getenv("SPEECH_REGION", "eastus")

st.set_page_config(
    page_title="Readify AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Keep state clean
st.session_state.setdefault("extracted_text", "")
st.session_state.setdefault("audio_bytes", b"")
st.session_state.setdefault("last_file_name", "")


# =========================================================
# HELPERS
# =========================================================

def has_azure_creds() -> bool:
    return all([VISION_KEY, VISION_ENDPOINT, SPEECH_KEY, SPEECH_REGION])


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    if not VISION_KEY or not VISION_ENDPOINT:
        raise RuntimeError("Vision credentials are missing.")

    client = ComputerVisionClient(
        VISION_ENDPOINT,
        CognitiveServicesCredentials(VISION_KEY),
    )

    image_stream = io.BytesIO(image_bytes)
    read_response = client.read_in_stream(image_stream, raw=True)

    operation_location = read_response.headers.get("Operation-Location")
    if not operation_location:
        raise RuntimeError("Azure OCR did not return an operation location.")

    operation_id = operation_location.split("/")[-1]

    timeout_seconds = 60
    start_time = time.time()

    while True:
        result = client.get_read_result(operation_id)

        if result.status not in ["notStarted", "running"]:
            break

        if time.time() - start_time > timeout_seconds:
            raise TimeoutError("OCR timed out. Try a clearer image or smaller file.")

        time.sleep(0.5)

    if result.status != OperationStatusCodes.succeeded:
        raise RuntimeError("OCR failed. Please try another image.")

    lines = []
    for page in result.analyze_result.read_results:
        for line in page.lines:
            cleaned = line.text.strip()
            if cleaned:
                lines.append(cleaned)

    return "\n".join(lines).strip()


def text_to_speech_bytes(text: str) -> bytes:
    if not SPEECH_KEY or not SPEECH_REGION:
        raise RuntimeError("Speech credentials are missing.")

    speech_config = SpeechConfig(
        subscription=SPEECH_KEY,
        region=SPEECH_REGION,
    )

    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    result = synthesizer.speak_text_async(text).get()

    if result.reason != ResultReason.SynthesizingAudioCompleted:
        if result.reason == ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise RuntimeError(
                f"Speech synthesis canceled: {cancellation.reason}. {cancellation.error_details or ''}"
            )
        raise RuntimeError("Speech synthesis failed.")

    return result.audio_data


def safe_filename(name: str) -> str:
    if not name:
        return "image"
    base = os.path.splitext(name)[0]
    return "".join(c for c in base if c.isalnum() or c in ("_", "-")).strip() or "image"


def clear_results():
    st.session_state["extracted_text"] = ""
    st.session_state["audio_bytes"] = b""


# =========================================================
# STYLES
# =========================================================

st.markdown(
    """
    <style>
    /* Hide Streamlit chrome */
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Nuke the Streamlit Cloud badge */
    a[href="https://streamlit.io/cloud"],
    a[href*="streamlit.io/cloud"],
    [class*="viewerBadge"],
    [class*="stBadge"],
    div[class*="gzau3"],
    div[class*="nim44"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        width: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
        z-index: -9999 !important;
    }
    </style>

    <script>
    // Remove Streamlit Cloud badge on load and whenever it gets re-injected
    function removeStreamlitBadge() {
        const badges = document.querySelectorAll(
            'a[href*="streamlit.io/cloud"], [class*="viewerBadge"], [class*="stBadge"]'
        );
        badges.forEach(el => el.remove());
    }

    // Run immediately
    removeStreamlitBadge();

    // Watch for dynamic injection
    const observer = new MutationObserver(() => removeStreamlitBadge());
    observer.observe(document.body, { childList: true, subtree: true });
    </script>

    <style>

    /* App background */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.18), transparent 25%),
            radial-gradient(circle at 85% 15%, rgba(168, 85, 247, 0.16), transparent 22%),
            linear-gradient(180deg, #050816 0%, #07111f 45%, #040711 100%);
        color: #e5eefb;
    }

    .block-container {
        padding-top: 0.7rem;
        padding-bottom: 0.8rem;
        max-width: 1180px;
    }

    /* Typography */
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        font-size: 0.84rem;
        font-weight: 600;
        margin-bottom: 0.65rem;
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 2.15rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0.05rem 0 0.4rem 0;
        color: #f8fbff;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        color: #94a3b8;
        margin-bottom: 0.85rem;
    }

    .mini-note {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1rem 1rem 1rem 1rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }

    .section-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #e8eef9;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }

    .subtle {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .stat-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.7rem;
        margin-top: 0.8rem;
    }

    .stat-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 0.75rem 0.8rem;
    }

    .stat-label {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-bottom: 0.25rem;
    }

    .stat-value {
        color: #f8fbff;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 14px;
        padding: 0.78rem 1rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.05);
        box-shadow: 0 16px 36px rgba(168, 85, 247, 0.28);
    }

    .stDownloadButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        color: #f8fbff;
        font-weight: 700;
        padding: 0.72rem 1rem;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 18px;
        border: 1px dashed rgba(255,255,255,0.13);
        padding: 0.2rem;
        background: rgba(255,255,255,0.025);
    }

    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none;
        padding: 0.45rem;
    }

    [data-testid="stFileUploaderDropzone"] > div {
        color: #dbe7f5 !important;
    }

    /* Preview image */
    img {
        border-radius: 16px !important;
    }

    /* Code / output */
    .output-wrap {
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.04);
        padding: 0.85rem;
        min-height: 200px;
        max-height: 320px;
        overflow: auto;
    }

    .output-pre {
        white-space: pre-wrap;
        word-break: break-word;
        color: #edf4ff;
        font-size: 0.95rem;
        line-height: 1.55;
        margin: 0;
    }

    /* Audio */
    [data-testid="stAudio"] {
        border-radius: 14px;
        overflow: hidden;
    }

    /* Make spacing compact */
    .element-container {
        margin-bottom: 0.35rem;
    }

    /* Reduce accidental empty gaps in some layouts */
    [data-testid="stHorizontalBlock"] {
        gap: 0.8rem;
    }

    /* Footer */
    .site-footer {
        margin-top: 3rem;
        padding: 1.5rem 0 1.2rem 0;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
        text-align: center;
    }

    .footer-name {
        font-size: 0.95rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.01em;
    }

    .footer-links {
        display: flex;
        justify-content: center;
        gap: 1.4rem;
        margin-top: 0.55rem;
        flex-wrap: wrap;
    }

    .footer-links a {
        color: #94a3b8;
        text-decoration: none;
        font-size: 0.82rem;
        font-weight: 500;
        transition: color 0.2s ease;
    }

    .footer-links a:hover {
        color: #a78bfa;
    }

    .footer-divider {
        color: rgba(255, 255, 255, 0.12);
        font-size: 0.7rem;
        user-select: none;
    }

    .footer-copy {
        color: #64748b;
        font-size: 0.74rem;
        margin-top: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown("<div class='hero-badge'>🧠 Azure OCR · 🎙️ Text to Speech</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-title'>Readify AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-subtitle'>Upload an image, extract the text, and turn it into clean speech with a polished one-screen workflow.</div>",
    unsafe_allow_html=True,
)

if not has_azure_creds():
    st.error(
        "Missing Azure credentials. Set `VISION_KEY`, `VISION_ENDPOINT`, `SPEECH_KEY`, and `SPEECH_REGION` in your environment."
    )


# =========================================================
# TOP LAYOUT
# =========================================================

left, right = st.columns([1.02, 1], gap="large")

with left:
    # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>1. Upload</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        label_visibility="collapsed",
        help="Best results come from clear, high-contrast images.",
    )

    btn_col_1, btn_col_2 = st.columns([1, 0.15], gap="small")
    with btn_col_1:
        run = st.button("✨ Extract Text & Generate Speech")

    with btn_col_2:
        st.markdown("", unsafe_allow_html=True)

    if uploaded:
        file_label = f"{uploaded.name} · {round(len(uploaded.getvalue()) / 1024, 1)} KB"
        st.markdown(f"<div class='mini-note'>Selected: {html.escape(file_label)}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='mini-note'>Choose an image to start.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>2. Preview</div>", unsafe_allow_html=True)

    if uploaded:
        st.image(uploaded, use_container_width=True)
    else:
        st.markdown(
            """
            <div style="
                min-height: 250px;
                display:flex;
                align-items:center;
                justify-content:center;
                border-radius:16px;
                border:1px dashed rgba(255,255,255,0.14);
                background: rgba(255,255,255,0.03);
                color:#94a3b8;
                font-size:0.95rem;
                text-align:center;
                padding:1rem;
            ">
                Your image preview will appear here.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PROCESSING
# =========================================================

if run:
    if not uploaded:
        st.warning("Please upload an image first.")
    elif not has_azure_creds():
        st.error("Azure keys are not configured.")
    else:
        try:
            with st.spinner("Reading image with Azure OCR..."):
                extracted = extract_text_from_image_bytes(uploaded.getvalue())

            if not extracted:
                clear_results()
                st.warning("No text detected. Try a clearer image or higher contrast.")
            else:
                with st.spinner("Generating speech..."):
                    audio_bytes = text_to_speech_bytes(extracted)

                st.session_state["extracted_text"] = extracted
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["last_file_name"] = uploaded.name
                st.toast("Done. Text extracted and audio generated.", icon="✅")

        except Exception as e:
            clear_results()
            st.error(f"Something went wrong: {e}")


# Auto-reuse results for the currently loaded file
if uploaded and st.session_state["last_file_name"] != uploaded.name:
    # keep results only if user is still on the same file
    pass


# =========================================================
# RESULTS
# =========================================================

if st.session_state["extracted_text"] or st.session_state["audio_bytes"]:
    st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>3. Result</div>", unsafe_allow_html=True)

    result_left, result_right = st.columns([1.25, 1], gap="large")

    with result_left:
        # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 📝 Extracted Text", unsafe_allow_html=True)

        text_value = st.session_state["extracted_text"].strip()
        if text_value:
            pretty_text = html.escape(text_value).replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="output-wrap">
                    <div class="output-pre">{pretty_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            word_count = len(text_value.split())
            char_count = len(text_value)
            st.markdown(
                f"""
                <div class="stat-row">
                    <div class="stat-box">
                        <div class="stat-label">Words</div>
                        <div class="stat-value">{word_count}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Characters</div>
                        <div class="stat-value">{char_count}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Status</div>
                        <div class="stat-value">Ready</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No extracted text yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    with result_right:
        # st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🔊 Audio Output", unsafe_allow_html=True)

        if st.session_state["audio_bytes"]:
            st.audio(st.session_state["audio_bytes"], format="audio/wav", autoplay=True)
            st.download_button(
                "⬇ Download Audio",
                data=st.session_state["audio_bytes"],
                file_name=f"{safe_filename(st.session_state['last_file_name'])}.wav",
                mime="audio/wav",
                width='stretch',
            )
        else:
            st.info("Audio will appear here after processing.")

        st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="site-footer">
        <div class="footer-name">Built by Priyanshi Chouhan</div>
        <div class="footer-copy">© 2026 Priyanshi Chouhan</div>
    </div>
    """,
    unsafe_allow_html=True,
)
