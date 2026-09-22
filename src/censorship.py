"""
censorship.py
--------------
STAGE 6 of the pipeline: Content Censorship Output.

Produces the three final deliverables described in the spec:
  1. A censored video: flagged audio spans are muted and replaced
     with a 1kHz beep tone (PyDub), then re-muxed with the original video.
  2. A clean .srt subtitle file with offensive words removed/masked.
  3. A PDF content report summarising all flagged content.
"""

import logging
import subprocess
from pathlib import Path
from typing import List

import srt
from datetime import timedelta
from pydub import AudioSegment
from pydub.generators import Sine
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

from config import CensorshipConfig
from transcription import TranscriptSegment
from fusion import TimelineEvent

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Audio censorship
# --------------------------------------------------------------------------

def censor_audio(
    audio_path: str,
    timeline: List[TimelineEvent],
    output_path: str,
    cfg: CensorshipConfig = CensorshipConfig(),
) -> str:
    """Replace flagged spans in the audio with a beep tone."""
    audio = AudioSegment.from_file(audio_path)

    for event in timeline:
        start_ms = max(0, int(event.start * 1000) - cfg.padding_ms)
        end_ms = min(len(audio), int(event.end * 1000) + cfg.padding_ms)
        duration_ms = end_ms - start_ms
        if duration_ms <= 0:
            continue

        beep = Sine(cfg.beep_freq_hz).to_audio_segment(duration=duration_ms).apply_gain(
            cfg.beep_gain_db
        )
        audio = audio[:start_ms] + beep + audio[end_ms:]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_path, format="wav")
    logger.info("Censored audio (%d spans muted) saved to %s", len(timeline), output_path)
    return output_path


def mux_video_with_censored_audio(
    original_video_path: str,
    censored_audio_path: str,
    output_video_path: str,
) -> str:
    """Replace the original video's audio track with the censored one."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original_video_path),
        "-i", str(censored_audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        str(output_video_path),
    ]
    logger.info("Muxing censored audio into video: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg mux failed:\n{result.stderr[-2000:]}")
    logger.info("Censored video saved to %s", output_video_path)
    return output_video_path


# --------------------------------------------------------------------------
# 2. Subtitle censorship
# --------------------------------------------------------------------------

def _mask_word(word: str) -> str:
    if len(word) <= 2:
        return "*" * len(word)
    return word[0] + "*" * (len(word) - 2) + word[-1]


def generate_clean_srt(
    segments: List[TranscriptSegment],
    timeline: List[TimelineEvent],
    output_path: str,
) -> str:
    """Generate an SRT file with flagged words masked out."""
    flagged_by_segment = {}
    for e in timeline:
        flagged_by_segment.setdefault(e.segment_text, set()).add(e.word.lower().strip())

    subs = []
    for i, seg in enumerate(segments, start=1):
        text = seg.text
        words_to_mask = flagged_by_segment.get(seg.text, set())
        if words_to_mask:
            tokens = text.split()
            tokens = [
                _mask_word(t) if t.strip(".,!?\"'").lower() in words_to_mask else t
                for t in tokens
            ]
            text = " ".join(tokens)

        subs.append(srt.Subtitle(
            index=i,
            start=timedelta(seconds=seg.start),
            end=timedelta(seconds=seg.end),
            content=text,
        ))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    logger.info("Clean SRT saved to %s", output_path)
    return output_path


# --------------------------------------------------------------------------
# 3. PDF content report
# --------------------------------------------------------------------------

def generate_pdf_report(
    timeline: List[TimelineEvent],
    output_path: str,
    source_filename: str = "",
) -> str:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Content Moderation Report", styles["Title"]))
    if source_filename:
        elements.append(Paragraph(f"Source file: {source_filename}", styles["Normal"]))
    elements.append(Paragraph(f"Total flagged instances: {len(timeline)}", styles["Normal"]))
    elements.append(Paragraph(
        f"Unique speakers implicated: {len({e.speaker for e in timeline})}", styles["Normal"]
    ))
    elements.append(Spacer(1, 16))

    table_data = [["Timestamp", "Speaker", "Word (masked)", "Severity", "Detection Source"]]
    for e in timeline:
        ts = f"{int(e.start // 60):02d}:{e.start % 60:05.2f}"
        table_data.append([ts, e.speaker, _mask_word(e.word), f"{e.severity:.2f}", e.source])

    table = Table(table_data, colWidths=[70, 70, 110, 60, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b52")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ]))
    elements.append(table)

    doc.build(elements)
    logger.info("PDF report saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse
    from fusion import load_timeline
    from transcription import load_transcript

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Produce censored outputs from a fused timeline.")
    parser.add_argument("--audio", required=True, help="Extracted WAV audio path")
    parser.add_argument("--video", required=False, help="Original video path (for re-muxing)")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    timeline = load_timeline(args.timeline)
    segments = load_transcript(args.transcript)

    censored_audio = censor_audio(args.audio, timeline, f"{args.outdir}/censored_audio.wav")
    generate_clean_srt(segments, timeline, f"{args.outdir}/clean_subtitles.srt")
    generate_pdf_report(timeline, f"{args.outdir}/content_report.pdf", source_filename=args.video or args.audio)

    if args.video:
        mux_video_with_censored_audio(args.video, censored_audio, f"{args.outdir}/censored_video.mp4")
