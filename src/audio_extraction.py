"""
audio_extraction.py
--------------------
STAGE 1 of the pipeline: Audio Extraction.

Extracts a mono, 16kHz WAV audio track from an input video file using
FFmpeg, then applies light preprocessing (loudness normalisation) to
improve downstream ASR quality.
"""

import subprocess
import logging
from pathlib import Path

from config import AudioConfig

logger = logging.getLogger(__name__)


class AudioExtractionError(RuntimeError):
    pass


def extract_audio(video_path: str, output_path: str, cfg: AudioConfig = AudioConfig()) -> str:
    """
    Extract and preprocess audio from a video file.

    Parameters
    ----------
    video_path : str
        Path to the source movie/video file (mp4, mkv, mov, etc.)
    output_path : str
        Destination .wav path.
    cfg : AudioConfig
        Sample rate / channel settings.

    Returns
    -------
    str : path to the extracted WAV file.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -ac  : audio channels (mono)
    # -ar  : sample rate
    # loudnorm : EBU R128 loudness normalisation filter (improves ASR robustness
    #            to quiet dialogue / inconsistent movie mixing levels)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # no video
        "-ac", str(cfg.channels),
        "-ar", str(cfg.sample_rate),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_path),
    ]

    logger.info("Extracting audio: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise AudioExtractionError(
            f"FFmpeg failed (code {result.returncode}):\n{result.stderr[-2000:]}"
        )

    if not output_path.exists():
        raise AudioExtractionError("FFmpeg reported success but no output file was created.")

    logger.info("Audio extracted to %s", output_path)
    return str(output_path)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract audio from a video file.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", default="work/audio.wav", help="Output WAV path")
    args = parser.parse_args()

    extract_audio(args.video, args.output)
