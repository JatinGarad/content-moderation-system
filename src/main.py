"""
main.py
-------
End-to-end pipeline orchestration, wiring together all six stages:

  1. audio_extraction.py  -> WAV audio
  2. transcription.py     -> timestamped transcript
  3. diarization.py       -> speaker segments
  4. profanity_detection.py -> flagged words
  5. fusion.py             -> attributed timeline
  6. censorship.py         -> censored video, clean SRT, PDF report

Usage:
    python main.py path/to/movie.mp4 --outdir output/
"""

import argparse
import logging
import time
from pathlib import Path

from config import PipelineConfig
from audio_extraction import extract_audio
from transcription import Transcriber, save_transcript
from diarization import Diarizer, save_diarization
from profanity_detection import ProfanityDetector
from fusion import fuse, save_timeline
from censorship import (
    censor_audio,
    generate_clean_srt,
    generate_pdf_report,
    mux_video_with_censored_audio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("pipeline")


def run_pipeline(video_path: str, cfg: PipelineConfig = None, num_speakers: int = None):
    cfg = cfg or PipelineConfig()
    cfg.ensure_dirs()

    t0 = time.time()
    video_path = Path(video_path)
    stem = video_path.stem

    # --- Stage 1: Audio Extraction ---
    logger.info("=== STAGE 1/6: Audio Extraction ===")
    audio_path = extract_audio(
        str(video_path), str(cfg.work_dir / f"{stem}_audio.wav"), cfg.audio
    )

    # --- Stage 2: ASR ---
    logger.info("=== STAGE 2/6: Automatic Speech Recognition ===")
    transcriber = Transcriber(cfg.asr)
    segments = transcriber.transcribe(audio_path)
    save_transcript(segments, str(cfg.work_dir / f"{stem}_transcript.json"))

    # --- Stage 3: Diarization ---
    logger.info("=== STAGE 3/6: Speaker Diarization ===")
    diarizer = Diarizer(cfg.diarization)
    diar_segments = diarizer.diarize(audio_path, num_speakers=num_speakers)
    save_diarization(diar_segments, str(cfg.work_dir / f"{stem}_diarization.json"))

    # --- Stage 4: Offensive Language Detection ---
    logger.info("=== STAGE 4/6: Offensive Language Detection ===")
    detector = ProfanityDetector(cfg.profanity)
    flagged = detector.detect(segments)

    # --- Stage 5: Multimodal Fusion ---
    logger.info("=== STAGE 5/6: Multimodal Fusion ===")
    timeline = fuse(segments, diar_segments, flagged)
    save_timeline(timeline, str(cfg.work_dir / f"{stem}_timeline.json"))

    # --- Stage 6: Content Censorship Output ---
    logger.info("=== STAGE 6/6: Content Censorship Output ===")
    censored_audio_path = censor_audio(
        audio_path, timeline, str(cfg.output_dir / f"{stem}_censored_audio.wav"), cfg.censorship
    )
    srt_path = generate_clean_srt(
        segments, timeline, str(cfg.output_dir / f"{stem}_clean.srt")
    )
    pdf_path = generate_pdf_report(
        timeline, str(cfg.output_dir / f"{stem}_report.pdf"), source_filename=str(video_path)
    )
    censored_video_path = mux_video_with_censored_audio(
        str(video_path), censored_audio_path, str(cfg.output_dir / f"{stem}_censored.mp4")
    )

    elapsed = time.time() - t0
    logger.info("Pipeline complete in %.1fs.", elapsed)
    logger.info("Outputs:\n  Censored video: %s\n  Clean subtitles: %s\n  Report: %s",
                censored_video_path, srt_path, pdf_path)

    return {
        "censored_video": censored_video_path,
        "clean_srt": srt_path,
        "report_pdf": pdf_path,
        "timeline": timeline,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full multimedia content moderation pipeline on a video file."
    )
    parser.add_argument("video", help="Path to input movie/video file")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--workdir", default="work", help="Intermediate working directory")
    parser.add_argument("--asr-model", default="large-v3", help="Whisper model size")
    parser.add_argument(
        "--profanity-method",
        choices=["rule_based", "roberta", "ensemble"],
        default="ensemble",
    )
    parser.add_argument("--num-speakers", type=int, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig()
    cfg.output_dir = Path(args.outdir)
    cfg.work_dir = Path(args.workdir)
    cfg.asr.model_name = args.asr_model
    cfg.profanity.method = args.profanity_method

    run_pipeline(args.video, cfg, num_speakers=args.num_speakers)
