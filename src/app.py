"""
app.py
------
Streamlit web interface for the content moderation pipeline
("Streamlit ... ideal for ML demo applications" per the spec's tech table).

Run with:
    streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig
from main import run_pipeline

st.set_page_config(page_title="Movie Content Moderation System", layout="wide")

st.title("🎬 AI Content Moderation System")
st.caption(
    "Automatic speech recognition, speaker diarization, and NLP-based "
    "profanity detection & censorship for movie content."
)

with st.sidebar:
    st.header("Settings")
    asr_model = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=4)
    profanity_method = st.selectbox("Detection method", ["ensemble", "rule_based", "roberta"], index=0)
    num_speakers = st.number_input("Number of speakers (optional, 0 = auto)", min_value=0, value=0)

uploaded_file = st.file_uploader("Upload a movie/video file", type=["mp4", "mkv", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Run Content Moderation Pipeline", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        cfg = PipelineConfig()
        cfg.asr.model_name = asr_model
        cfg.profanity.method = profanity_method

        progress = st.empty()
        progress.info("Running pipeline — this can take several minutes depending on video length...")

        try:
            results = run_pipeline(
                tmp_path, cfg, num_speakers=(num_speakers or None)
            )
            progress.success("Pipeline complete!")

            st.subheader("Results")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Censored Video**")
                st.video(results["censored_video"])

            with col2:
                st.markdown("**Flagged Content Timeline**")
                st.dataframe(
                    [
                        {
                            "Time (s)": round(e.start, 2),
                            "Speaker": e.speaker,
                            "Severity": round(e.severity, 2),
                            "Source": e.source,
                        }
                        for e in results["timeline"]
                    ]
                )

            st.markdown("**Downloads**")
            with open(results["clean_srt"], "rb") as f:
                st.download_button("Download clean subtitles (.srt)", f, file_name="clean_subtitles.srt")
            with open(results["report_pdf"], "rb") as f:
                st.download_button("Download content report (.pdf)", f, file_name="content_report.pdf")

        except Exception as e:
            progress.error(f"Pipeline failed: {e}")
            st.exception(e)
