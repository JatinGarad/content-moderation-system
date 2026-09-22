"""
dashboard.py
------------
Interactive Streamlit interface and analytics dashboard for the
multimodal content moderation pipeline.

Replaces the basic single-screen app.py with a four-page application:

  1. Process      -- upload a video, configure, run the pipeline with
                     live per-stage progress reporting
  2. Results      -- interactive exploration of a completed run:
                     timeline chart, highlighted transcript, speaker
                     breakdown, severity distribution, media players
  3. Compare      -- run all three detection methods over the same
                     transcript and compare their outputs directly
                     (produces the rule-based vs transformer evidence
                     the dissertation's research question requires)
  4. Evaluation   -- metrics against the specification's target table,
                     with pass/fail status and run history

Run with:
    cd src
    streamlit run dashboard.py

Requires the same environment as main.py (Whisper, PyAnnote, FFmpeg,
HF_TOKEN). Altair is used for charting -- it ships with Streamlit, so
no additional dependency is introduced.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, ASRConfig, DiarizationConfig, ProfanityConfig, CensorshipConfig
from audio_extraction import extract_audio
from transcription import Transcriber, save_transcript, load_transcript
from diarization import Diarizer, save_diarization, load_diarization
from profanity_detection import ProfanityDetector
from fusion import fuse, save_timeline, load_timeline
from censorship import (
    censor_audio,
    generate_clean_srt,
    generate_pdf_report,
    mux_video_with_censored_audio,
)

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------
# Page config and styling
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Content Moderation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stMetric { background: #f7f7fa; border: 1px solid #e4e4ec;
                  border-radius: 8px; padding: 12px 14px; }
      .transcript-box { background: #ffffff; border: 1px solid #e4e4ec;
                        border-radius: 8px; padding: 18px 22px;
                        line-height: 2.1; font-size: 15px; }
      .flagged { background: #ffd9d9; color: #8a1c1c; font-weight: 600;
                 border-radius: 4px; padding: 2px 5px;
                 border-bottom: 2px solid #c53030; }
      .spk-tag { display: inline-block; background: #eef1ff; color: #33409e;
                 border-radius: 4px; padding: 1px 7px; font-size: 12px;
                 font-weight: 600; margin-right: 8px; }
      .warn-banner { background: #fff5e6; border-left: 4px solid #e8a33d;
                     padding: 12px 16px; border-radius: 4px; margin: 8px 0 16px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

RUN_HISTORY = Path("work/run_history.json")
SPEC_TARGETS = {
    "wer": ("ASR Word Error Rate", 0.10, "lower"),
    "der": ("Diarization Error Rate", 0.15, "lower"),
    "f1": ("Profanity Detection F1", 0.90, "higher"),
    "end_to_end_accuracy": ("End-to-End Detection Accuracy", 0.85, "higher"),
    "processing_time_ratio": ("Processing Time (× real-time)", 2.0, "lower"),
    "sus_score": ("Usability (SUS Score)", 70.0, "higher"),
}


# --------------------------------------------------------------------------
# Cached resources -- models are expensive; never reload them on a rerun
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_transcriber(model_name: str):
    return Transcriber(ASRConfig(model_name=model_name))


@st.cache_resource(show_spinner=False)
def get_diarizer():
    return Diarizer(DiarizationConfig())


@st.cache_resource(show_spinner=False)
def get_detector(method: str, threshold: float, model_dir: str):
    return ProfanityDetector(
        ProfanityConfig(method=method, severity_threshold=threshold,
                        roberta_model_dir=model_dir)
    )


def roberta_is_finetuned(model_dir: str) -> bool:
    """True only if a real fine-tuned checkpoint exists on disk."""
    p = Path(model_dir)
    if not p.exists():
        return False
    return any((p / f).exists() for f in
               ("model.safetensors", "pytorch_model.bin", "tf_model.h5"))


# --------------------------------------------------------------------------
# Run history persistence
# --------------------------------------------------------------------------

def load_history() -> list[dict]:
    if RUN_HISTORY.exists():
        try:
            return json.loads(RUN_HISTORY.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def append_history(entry: dict):
    hist = load_history()
    hist.append(entry)
    RUN_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    RUN_HISTORY.write_text(json.dumps(hist, indent=2))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h else f"{int(m):02d}:{s:05.2f}"


def timeline_to_df(timeline) -> pd.DataFrame:
    if not timeline:
        return pd.DataFrame(
            columns=["start", "end", "word", "speaker", "severity", "source", "segment_text"]
        )
    return pd.DataFrame([asdict(e) if not isinstance(e, dict) else e for e in timeline])


def mask(word: str) -> str:
    w = word.strip()
    if len(w) <= 2:
        return "*" * len(w)
    return w[0] + "*" * (len(w) - 2) + w[-1]


def render_transcript(segments, timeline, show_masked: bool = True) -> str:
    """Build an HTML transcript with flagged words highlighted in place."""
    flagged_spans = [(e.start, e.end) if not isinstance(e, dict)
                     else (e["start"], e["end"]) for e in timeline]

    def is_flagged(w) -> bool:
        return any(abs(w.start - s) < 1e-6 and abs(w.end - en) < 1e-6
                   for s, en in flagged_spans)

    html = []
    for seg in segments:
        html.append(f'<div style="margin-bottom:14px;">'
                    f'<span class="spk-tag">{fmt_ts(seg.start)}</span>')
        if seg.words:
            for w in seg.words:
                token = mask(w.word) if (show_masked and is_flagged(w)) else w.word
                if is_flagged(w):
                    html.append(f'<span class="flagged">{token}</span> ')
                else:
                    html.append(f"{token} ")
        else:
            html.append(seg.text)
        html.append("</div>")
    return f'<div class="transcript-box">{"".join(html)}</div>'


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

st.sidebar.title("🎬 Content Moderation")
st.sidebar.caption("ASR · Diarization · NLP")

page = st.sidebar.radio(
    "Navigate",
    ["Process", "Results", "Compare Methods", "Evaluation"],
    label_visibility="collapsed",
)

st.sidebar.divider()

if "run" in st.session_state:
    r = st.session_state["run"]
    st.sidebar.success(f"Loaded run: **{r['stem']}**")
    st.sidebar.caption(f"{len(r['timeline'])} flagged · {r['elapsed']:.1f}s")
else:
    st.sidebar.info("No run loaded yet.")


# ==========================================================================
# PAGE 1 -- PROCESS
# ==========================================================================

if page == "Process":
    st.title("Process Video")
    st.caption(
        "Upload a video file and run the six-stage moderation pipeline. "
        "Each stage reports progress as it completes."
    )

    cfg = PipelineConfig()

    with st.sidebar:
        st.subheader("Pipeline settings")
        asr_model = st.selectbox(
            "Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=4,
            help="Larger models are more accurate but substantially slower on CPU.",
        )
        method = st.selectbox(
            "Detection method", ["ensemble", "rule_based", "roberta"], index=0,
        )
        threshold = st.slider(
            "Detection sensitivity (confidence threshold)", 0.0, 1.0, 0.5, 0.05,
            help="Lower values flag more words (higher recall, lower precision).",
        )
        num_speakers = st.number_input(
            "Known number of speakers (0 = auto-detect)", min_value=0, value=0,
        )
        st.subheader("Censorship")
        beep_hz = st.number_input("Beep frequency (Hz)", 200, 4000, 1000, 100)
        padding_ms = st.number_input("Padding around beep (ms)", 0, 500, 40, 10)
        model_dir = st.text_input("RoBERTa checkpoint directory", "models/roberta-profanity")

    if method in ("roberta", "ensemble") and not roberta_is_finetuned(model_dir):
        st.markdown(
            '<div class="warn-banner"><b>No fine-tuned RoBERTa checkpoint found at '
            f'<code>{model_dir}</code>.</b><br>The classifier will fall back to base '
            '<code>roberta-base</code> with a randomly initialised classification head. '
            'Its confidence scores will be meaningless. Run <code>train_roberta.py</code> '
            'before using any RoBERTa results in your dissertation.</div>',
            unsafe_allow_html=True,
        )

    uploaded = st.file_uploader(
        "Video file", type=["mp4", "mkv", "mov", "avi", "m4v", "webm"]
    )

    col_a, col_b = st.columns([2, 3])
    if uploaded is not None:
        with col_a:
            st.video(uploaded)
        with col_b:
            st.write("**File**", uploaded.name)
            st.write("**Size**", f"{uploaded.size / 1e6:.1f} MB")
            st.write("**Method**", method)
            st.write("**ASR model**", asr_model)

        if st.button("▶ Run pipeline", type="primary", use_container_width=True):
            cfg.asr.model_name = asr_model
            cfg.profanity.method = method
            cfg.profanity.severity_threshold = threshold
            cfg.profanity.roberta_model_dir = model_dir
            cfg.censorship.beep_freq_hz = int(beep_hz)
            cfg.censorship.padding_ms = int(padding_ms)
            cfg.ensure_dirs()

            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                video_path = tmp.name

            stem = Path(uploaded.name).stem
            t0 = time.time()
            stage_times = {}

            try:
                with st.status("Running pipeline…", expanded=True) as status:
                    # Stage 1
                    st.write("**Stage 1/6** — Audio extraction")
                    ts = time.time()
                    audio_path = extract_audio(
                        video_path, str(cfg.work_dir / f"{stem}_audio.wav"), cfg.audio
                    )
                    stage_times["audio_extraction"] = time.time() - ts
                    st.write(f"↳ {audio_path}  ·  {stage_times['audio_extraction']:.1f}s")

                    # Stage 2
                    st.write(f"**Stage 2/6** — Speech recognition (Whisper `{asr_model}`)")
                    ts = time.time()
                    transcriber = get_transcriber(asr_model)
                    segments = transcriber.transcribe(audio_path)
                    save_transcript(segments, str(cfg.work_dir / f"{stem}_transcript.json"))
                    stage_times["transcription"] = time.time() - ts
                    n_words = sum(len(s.words) for s in segments)
                    st.write(f"↳ {len(segments)} segments, {n_words} words  ·  "
                             f"{stage_times['transcription']:.1f}s")

                    # Stage 3
                    st.write("**Stage 3/6** — Speaker diarization (PyAnnote)")
                    ts = time.time()
                    diarizer = get_diarizer()
                    diar = diarizer.diarize(audio_path,
                                            num_speakers=(num_speakers or None))
                    save_diarization(diar, str(cfg.work_dir / f"{stem}_diarization.json"))
                    stage_times["diarization"] = time.time() - ts
                    st.write(f"↳ {len(diar)} turns, {len({d.speaker for d in diar})} "
                             f"speakers  ·  {stage_times['diarization']:.1f}s")

                    # Stage 4
                    st.write(f"**Stage 4/6** — Offensive language detection (`{method}`)")
                    ts = time.time()
                    detector = get_detector(method, threshold, model_dir)
                    flagged = detector.detect(segments)
                    stage_times["detection"] = time.time() - ts
                    st.write(f"↳ {len(flagged)} words flagged  ·  "
                             f"{stage_times['detection']:.1f}s")

                    # Stage 5
                    st.write("**Stage 5/6** — Multimodal fusion")
                    ts = time.time()
                    timeline = fuse(segments, diar, flagged)
                    save_timeline(timeline, str(cfg.work_dir / f"{stem}_timeline.json"))
                    stage_times["fusion"] = time.time() - ts
                    unknown = sum(1 for e in timeline if e.speaker == "UNKNOWN")
                    st.write(f"↳ {len(timeline)} events attributed"
                             + (f" ({unknown} unattributed)" if unknown else "")
                             + f"  ·  {stage_times['fusion']:.1f}s")

                    # Stage 6
                    st.write("**Stage 6/6** — Censorship output")
                    ts = time.time()
                    censored_audio = censor_audio(
                        audio_path, timeline,
                        str(cfg.output_dir / f"{stem}_censored_audio.wav"), cfg.censorship)
                    srt_path = generate_clean_srt(
                        segments, timeline, str(cfg.output_dir / f"{stem}_clean.srt"))
                    pdf_path = generate_pdf_report(
                        timeline, str(cfg.output_dir / f"{stem}_report.pdf"),
                        source_filename=uploaded.name)
                    try:
                        video_out = mux_video_with_censored_audio(
                            video_path, censored_audio,
                            str(cfg.output_dir / f"{stem}_censored.mp4"))
                    except RuntimeError as exc:
                        video_out = None
                        st.warning(f"Video mux failed (audio outputs are still valid): {exc}")
                    stage_times["censorship"] = time.time() - ts
                    st.write(f"↳ outputs written  ·  {stage_times['censorship']:.1f}s")

                    status.update(label="Pipeline complete", state="complete",
                                  expanded=False)

                elapsed = time.time() - t0
                duration = max(s.end for s in segments) if segments else 0.0

                st.session_state["run"] = {
                    "stem": stem,
                    "source_name": uploaded.name,
                    "segments": segments,
                    "diarization": diar,
                    "timeline": timeline,
                    "audio_path": audio_path,
                    "censored_audio": censored_audio,
                    "censored_video": video_out,
                    "srt_path": srt_path,
                    "pdf_path": pdf_path,
                    "elapsed": elapsed,
                    "duration": duration,
                    "stage_times": stage_times,
                    "method": method,
                    "asr_model": asr_model,
                    "threshold": threshold,
                }

                append_history({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "file": uploaded.name,
                    "asr_model": asr_model,
                    "method": method,
                    "threshold": threshold,
                    "n_segments": len(segments),
                    "n_words": n_words,
                    "n_speakers": len({d.speaker for d in diar}),
                    "n_flagged": len(timeline),
                    "duration_s": round(duration, 2),
                    "elapsed_s": round(elapsed, 2),
                    "rtf": round(elapsed / duration, 3) if duration else None,
                    "stage_times": {k: round(v, 2) for k, v in stage_times.items()},
                })

                st.success(
                    f"Done in {elapsed:.1f}s — {len(timeline)} flagged instance(s). "
                    "Open the **Results** page to explore."
                )
                if not timeline:
                    st.info(
                        "No offensive language was detected. If you expected detections, "
                        "check that the clip actually contains profanity and that your "
                        "detection method and threshold are set appropriately."
                    )

            except Exception as exc:  # noqa: BLE001
                st.error(f"Pipeline failed: {exc}")
                st.exception(exc)
    else:
        st.info("Upload a video file to begin.")


# ==========================================================================
# PAGE 2 -- RESULTS
# ==========================================================================

elif page == "Results":
    st.title("Results")

    if "run" not in st.session_state:
        st.warning("No run loaded. Process a video first, or load saved artefacts below.")
        with st.expander("Load a previous run from work/ artefacts"):
            stem_in = st.text_input("File stem (e.g. IMG_6087)")
            if st.button("Load") and stem_in:
                try:
                    segs = load_transcript(f"work/{stem_in}_transcript.json")
                    diar = load_diarization(f"work/{stem_in}_diarization.json")
                    tl = load_timeline(f"work/{stem_in}_timeline.json")
                    st.session_state["run"] = {
                        "stem": stem_in, "source_name": stem_in,
                        "segments": segs, "diarization": diar, "timeline": tl,
                        "audio_path": f"work/{stem_in}_audio.wav",
                        "censored_audio": f"output/{stem_in}_censored_audio.wav",
                        "censored_video": f"output/{stem_in}_censored.mp4",
                        "srt_path": f"output/{stem_in}_clean.srt",
                        "pdf_path": f"output/{stem_in}_report.pdf",
                        "elapsed": 0.0,
                        "duration": max((s.end for s in segs), default=0.0),
                        "stage_times": {}, "method": "—", "asr_model": "—",
                        "threshold": 0.5,
                    }
                    st.rerun()
                except (OSError, json.JSONDecodeError) as exc:
                    st.error(f"Could not load artefacts: {exc}")
        st.stop()

    run = st.session_state["run"]
    df = timeline_to_df(run["timeline"])
    segments = run["segments"]
    diar = run["diarization"]

    # --- Headline metrics -------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Flagged instances", len(df))
    c2.metric("Speakers detected", len({d.speaker for d in diar}))
    c3.metric("Clip duration", f"{run['duration']:.1f}s")
    c4.metric("Processing time", f"{run['elapsed']:.1f}s")
    rtf = run["elapsed"] / run["duration"] if run["duration"] else 0
    c5.metric("Real-time factor", f"{rtf:.2f}×",
              delta="within target" if rtf <= 2 else "over target",
              delta_color="normal" if rtf <= 2 else "inverse")

    st.divider()

    if df.empty:
        st.info(
            "This run produced no flagged content, so there is nothing to plot. "
            "The transcript is shown below for reference."
        )
    else:
        # --- Timeline chart ----------------------------------------------
        st.subheader("Flagged content over time")
        chart_df = df.copy()
        chart_df["label"] = chart_df["word"].apply(mask)
        chart_df["ts"] = chart_df["start"].apply(fmt_ts)

        timeline_chart = (
            alt.Chart(chart_df)
            .mark_circle(size=180, opacity=0.85)
            .encode(
                x=alt.X("start:Q", title="Time (seconds)",
                        scale=alt.Scale(domain=[0, run["duration"] or 1])),
                y=alt.Y("speaker:N", title="Speaker"),
                color=alt.Color("severity:Q", title="Severity",
                                scale=alt.Scale(scheme="reds", domain=[0, 1])),
                size=alt.Size("severity:Q", legend=None,
                              scale=alt.Scale(range=[80, 320])),
                tooltip=["ts", "label", "speaker", "severity", "source", "segment_text"],
            )
            .properties(height=max(160, 60 * chart_df["speaker"].nunique()))
            .interactive()
        )
        st.altair_chart(timeline_chart, use_container_width=True)

        # --- Breakdown charts --------------------------------------------
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("By speaker")
            spk = df.groupby("speaker").size().reset_index(name="count")
            st.altair_chart(
                alt.Chart(spk).mark_bar().encode(
                    x=alt.X("count:Q", title="Flagged words"),
                    y=alt.Y("speaker:N", title=None, sort="-x"),
                    color=alt.Color("speaker:N", legend=None,
                                    scale=alt.Scale(scheme="blues")),
                    tooltip=["speaker", "count"],
                ).properties(height=240),
                use_container_width=True,
            )
        with cc2:
            st.subheader("By detection source")
            src = df.groupby("source").size().reset_index(name="count")
            st.altair_chart(
                alt.Chart(src).mark_arc(innerRadius=55).encode(
                    theta="count:Q",
                    color=alt.Color("source:N", title="Source"),
                    tooltip=["source", "count"],
                ).properties(height=240),
                use_container_width=True,
            )

        st.subheader("Severity distribution")
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X("severity:Q", bin=alt.Bin(maxbins=20), title="Severity score"),
                y=alt.Y("count()", title="Count"),
                color=alt.value("#c53030"),
            ).properties(height=200),
            use_container_width=True,
        )

        # --- Filterable table --------------------------------------------
        st.subheader("Flagged instances")
        fc1, fc2 = st.columns([1, 1])
        speakers = fc1.multiselect("Filter by speaker",
                                   sorted(df["speaker"].unique()),
                                   default=sorted(df["speaker"].unique()))
        min_sev = fc2.slider("Minimum severity", 0.0, 1.0, 0.0, 0.05)

        view = df[df["speaker"].isin(speakers) & (df["severity"] >= min_sev)].copy()
        view["timestamp"] = view["start"].apply(fmt_ts)
        view["word"] = view["word"].apply(mask)
        st.dataframe(
            view[["timestamp", "speaker", "word", "severity", "source", "segment_text"]],
            use_container_width=True, hide_index=True,
        )

    # --- Transcript -------------------------------------------------------
    st.divider()
    st.subheader("Transcript")
    show_masked = st.toggle("Mask flagged words", value=True)
    st.markdown(render_transcript(segments, run["timeline"], show_masked),
                unsafe_allow_html=True)

    # --- Media and downloads ---------------------------------------------
    st.divider()
    st.subheader("Output media")
    m1, m2 = st.columns(2)
    with m1:
        st.caption("Original audio")
        if Path(run["audio_path"]).exists():
            st.audio(run["audio_path"])
    with m2:
        st.caption("Censored audio")
        if run["censored_audio"] and Path(run["censored_audio"]).exists():
            st.audio(run["censored_audio"])

    if run.get("censored_video") and Path(run["censored_video"]).exists():
        st.caption("Censored video")
        st.video(run["censored_video"])

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    for col, key, label, fname, mime in [
        (d1, "censored_video", "Censored video (.mp4)", "censored.mp4", "video/mp4"),
        (d2, "srt_path", "Clean subtitles (.srt)", "clean.srt", "text/plain"),
        (d3, "pdf_path", "Content report (.pdf)", "report.pdf", "application/pdf"),
    ]:
        path = run.get(key)
        if path and Path(path).exists():
            with open(path, "rb") as fh:
                col.download_button(label, fh, file_name=fname, mime=mime,
                                    use_container_width=True)
        else:
            col.button(label, disabled=True, use_container_width=True)


# ==========================================================================
# PAGE 3 -- COMPARE METHODS
# ==========================================================================

elif page == "Compare Methods":
    st.title("Detection Method Comparison")
    st.caption(
        "Runs the rule-based filter, the RoBERTa classifier, and the ensemble "
        "over the same transcript, so their outputs can be compared directly. "
        "This is the evidence your research question asks for."
    )

    if "run" not in st.session_state:
        st.warning("Process a video first — the comparison reuses its transcript.")
        st.stop()

    run = st.session_state["run"]
    segments = run["segments"]
    diar = run["diarization"]

    model_dir = st.text_input("RoBERTa checkpoint directory", "models/roberta-profanity")
    if not roberta_is_finetuned(model_dir):
        st.markdown(
            '<div class="warn-banner"><b>RoBERTa is not fine-tuned.</b> '
            'The comparison below will run, but the transformer column reflects an '
            'untrained classification head and must not be reported as a result.</div>',
            unsafe_allow_html=True,
        )

    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

    if st.button("Run comparison", type="primary"):
        results = {}
        prog = st.progress(0.0)
        for i, m in enumerate(["rule_based", "roberta", "ensemble"]):
            try:
                det = get_detector(m, threshold, model_dir)
                flagged = det.detect(segments)
                results[m] = fuse(segments, diar, flagged)
            except Exception as exc:  # noqa: BLE001
                st.error(f"`{m}` failed: {exc}")
                results[m] = []
            prog.progress((i + 1) / 3)
        st.session_state["comparison"] = results

    if "comparison" in st.session_state:
        results = st.session_state["comparison"]

        cols = st.columns(3)
        for col, (m, tl) in zip(cols, results.items()):
            col.metric(m.replace("_", " ").title(), len(tl), "flagged words")

        summary = pd.DataFrame([
            {
                "Method": m.replace("_", " ").title(),
                "Flagged": len(tl),
                "Unique words": len({e.word.lower().strip() for e in tl}),
                "Mean severity": round(
                    sum(e.severity for e in tl) / len(tl), 3) if tl else 0.0,
                "Speakers implicated": len({e.speaker for e in tl}),
            }
            for m, tl in results.items()
        ])
        st.subheader("Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Set overlap
        sets = {m: {(round(e.start, 3), e.word.lower().strip()) for e in tl}
                for m, tl in results.items()}
        st.subheader("Agreement between methods")
        rb, rob = sets["rule_based"], sets["roberta"]
        o1, o2, o3 = st.columns(3)
        o1.metric("Rule-based only", len(rb - rob))
        o2.metric("Both agree", len(rb & rob))
        o3.metric("RoBERTa only", len(rob - rb))

        if len(rb - rob) or len(rob - rb):
            st.caption("Disagreements — the interesting cases for your discussion chapter")
            rows = []
            for key in sorted(rb ^ rob):
                t, w = key
                rows.append({
                    "Time": fmt_ts(t),
                    "Word": mask(w),
                    "Flagged by": "rule-based only" if key in rb else "RoBERTa only",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Comparison chart
        chart_rows = []
        for m, tl in results.items():
            for e in tl:
                chart_rows.append({"method": m, "start": e.start,
                                   "severity": e.severity, "word": mask(e.word)})
        if chart_rows:
            cdf = pd.DataFrame(chart_rows)
            st.altair_chart(
                alt.Chart(cdf).mark_circle(size=140, opacity=0.8).encode(
                    x=alt.X("start:Q", title="Time (seconds)"),
                    y=alt.Y("method:N", title=None),
                    color=alt.Color("severity:Q", scale=alt.Scale(scheme="reds",
                                                                  domain=[0, 1])),
                    tooltip=["method", "word", "severity"],
                ).properties(height=180).interactive(),
                use_container_width=True,
            )

        csv = summary.to_csv(index=False).encode()
        st.download_button("Download comparison as CSV", csv,
                           file_name="method_comparison.csv", mime="text/csv")


# ==========================================================================
# PAGE 4 -- EVALUATION
# ==========================================================================

elif page == "Evaluation":
    st.title("Evaluation Dashboard")
    st.caption("System performance against the targets defined in the project specification.")

    st.subheader("Enter measured results")
    st.caption(
        "These must come from your annotated test set — computed with "
        "`evaluation.py`, or with `pyannote.metrics` for the DER you report formally."
    )

    e1, e2, e3 = st.columns(3)
    with e1:
        wer = st.number_input("ASR Word Error Rate", 0.0, 1.0, 0.0, 0.005, format="%.3f")
        der = st.number_input("Diarization Error Rate", 0.0, 1.0, 0.0, 0.005, format="%.3f")
    with e2:
        f1 = st.number_input("Profanity Detection F1", 0.0, 1.0, 0.0, 0.005, format="%.3f")
        acc = st.number_input("End-to-end Detection Accuracy", 0.0, 1.0, 0.0, 0.005,
                              format="%.3f")
    with e3:
        rtf_in = st.number_input("Processing time (× real-time)", 0.0, 100.0, 0.0, 0.1)
        sus = st.number_input("SUS score", 0.0, 100.0, 0.0, 1.0)

    measured = {"wer": wer, "der": der, "f1": f1, "end_to_end_accuracy": acc,
                "processing_time_ratio": rtf_in, "sus_score": sus}
    entered = {k: v for k, v in measured.items() if v > 0}

    if entered:
        rows = []
        for key, (label, target, direction) in SPEC_TARGETS.items():
            if key not in entered:
                continue
            value = entered[key]
            passed = value <= target if direction == "lower" else value >= target
            rows.append({"Metric": label, "Result": value, "Target": target,
                         "Status": "PASS" if passed else "FAIL"})
        res_df = pd.DataFrame(rows)

        st.subheader("Against specification targets")

        def _style_status(v):
            if v == "PASS":
                return "background-color:#d8f5d8;color:#12610f;font-weight:600"
            if v == "FAIL":
                return "background-color:#ffdada;color:#8a1c1c;font-weight:600"
            return ""

        try:
            styler = res_df.style
            # pandas >= 2.1 renamed Styler.applymap to Styler.map
            apply_fn = getattr(styler, "map", None) or styler.applymap
            st.dataframe(apply_fn(_style_status, subset=["Status"]),
                         use_container_width=True, hide_index=True)
        except Exception:  # noqa: BLE001 -- styling is cosmetic only
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Measured value vs target, as two layered marks (Altair 4/5 safe)
        bars = alt.Chart(res_df).mark_bar().encode(
            x=alt.X("Result:Q", title="Measured value"),
            y=alt.Y("Metric:N", title=None, sort="-x"),
            color=alt.Color("Status:N", scale=alt.Scale(
                domain=["PASS", "FAIL"], range=["#2f855a", "#c53030"])),
            tooltip=["Metric", "Result", "Target", "Status"],
        )
        ticks = alt.Chart(res_df).mark_tick(
            color="#333", thickness=3, size=26,
        ).encode(
            x=alt.X("Target:Q"),
            y=alt.Y("Metric:N", sort="-x"),
            tooltip=["Metric", "Target"],
        )
        st.altair_chart((bars + ticks).properties(height=320),
                        use_container_width=True)
        st.caption("Bars show your measured value; the black tick marks the specification target.")
        st.download_button("Download results as CSV",
                           res_df.to_csv(index=False).encode(),
                           file_name="evaluation_results.csv", mime="text/csv")
    else:
        st.info("Enter at least one measured value above to see the comparison.")

    st.divider()
    st.subheader("Confusion matrix calculator")
    m1, m2, m3 = st.columns(3)
    tp = m1.number_input("True positives", 0, 100000, 0)
    fp = m2.number_input("False positives", 0, 100000, 0)
    fn = m3.number_input("False negatives", 0, 100000, 0)
    if tp or fp or fn:
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1c = 2 * precision * recall / max(1e-9, precision + recall)
        p1, p2, p3 = st.columns(3)
        p1.metric("Precision", f"{precision:.3f}")
        p2.metric("Recall", f"{recall:.3f}")
        p3.metric("F1", f"{f1c:.3f}")

    st.divider()
    st.subheader("Run history")
    hist = load_history()
    if hist:
        hdf = pd.DataFrame(hist).drop(columns=["stage_times"], errors="ignore")
        st.dataframe(hdf, use_container_width=True, hide_index=True)

        if "rtf" in hdf.columns and hdf["rtf"].notna().any():
            st.caption("Real-time factor across runs")
            st.altair_chart(
                alt.Chart(hdf.dropna(subset=["rtf"])).mark_line(point=True).encode(
                    x=alt.X("timestamp:N", title=None,
                            axis=alt.Axis(labelAngle=-40)),
                    y=alt.Y("rtf:Q", title="× real-time"),
                    tooltip=["timestamp", "file", "asr_model", "rtf"],
                ).properties(height=220),
                use_container_width=True,
            )
        st.download_button("Download run history as CSV",
                           hdf.to_csv(index=False).encode(),
                           file_name="run_history.csv", mime="text/csv")
    else:
        st.info("No runs recorded yet. Process a video to start building history.")
