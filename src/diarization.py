"""
diarization.py
---------------
STAGE 3 of the pipeline: Speaker Diarization.

Uses PyAnnote Audio 3.1 to answer "who spoke when", producing a list of
speaker-labelled time segments. This is later merged with the ASR
transcript (Stage 5) to attribute offensive words to specific speakers.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
from pyannote.audio import Pipeline

from config import DiarizationConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float


class Diarizer:
    def __init__(self, cfg: DiarizationConfig = DiarizationConfig()):
        self.cfg = cfg
        # Use an explicit HF_TOKEN if set, otherwise fall back to the
        # credentials cached by `hf auth login` (huggingface_hub picks
        # these up automatically when token=True is passed).
        auth_token = cfg.hf_token if cfg.hf_token else True

        logger.info("Loading PyAnnote pipeline '%s' ...", cfg.model_name)
        try:
            # pyannote.audio >= 3.1 uses `token=`
            self.pipeline = Pipeline.from_pretrained(cfg.model_name, token=auth_token)
        except TypeError:
            # older pyannote.audio versions use `use_auth_token=`
            self.pipeline = Pipeline.from_pretrained(cfg.model_name, use_auth_token=auth_token)

        device = cfg.device if torch.cuda.is_available() else "cpu"
        if device != cfg.device:
            logger.warning("CUDA not available; PyAnnote will run on CPU (slower).")
        self.pipeline.to(torch.device(device))

    def diarize(self, audio_path: str, num_speakers: int = None) -> List[SpeakerSegment]:
        """
        Run diarization on the given audio file.

        Parameters
        ----------
        audio_path : str
        num_speakers : int, optional
            If known in advance, constrains the pipeline (improves accuracy).

        Returns
        -------
        List[SpeakerSegment]
        """
        logger.info("Running diarization on %s ...", audio_path)
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        result = self.pipeline(audio_path, **kwargs)

        # Newer pyannote.audio versions (community pipelines) wrap the result in a
        # DiarizeOutput object; older versions return an Annotation directly.
        if hasattr(result, "itertracks"):
            annotation = result
        elif hasattr(result, "speaker_diarization"):
            annotation = result.speaker_diarization
        elif hasattr(result, "exclusive_speaker_diarization"):
            annotation = result.exclusive_speaker_diarization
        else:
            raise AttributeError(
                f"Unrecognized diarization output type: {type(result)}. "
                f"Available attributes: {[a for a in dir(result) if not a.startswith('_')]}"
            )

        segments = [
            SpeakerSegment(speaker=speaker, start=turn.start, end=turn.end)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        logger.info("Diarization complete: %d speaker turns, %d unique speakers",
                     len(segments), len({s.speaker for s in segments}))
        return segments


def save_diarization(segments: List[SpeakerSegment], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in segments], f, indent=2)
    logger.info("Diarization saved to %s", out_path)


def load_diarization(path: str) -> List[SpeakerSegment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [SpeakerSegment(**d) for d in data]


def speaker_at(segments: List[SpeakerSegment], t: float) -> str:
    """Return the speaker label active at time t (seconds), or 'UNKNOWN'."""
    for seg in segments:
        if seg.start <= t <= seg.end:
            return seg.speaker
    return "UNKNOWN"


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Diarize speakers in an audio file.")
    parser.add_argument("audio", help="Path to input WAV file")
    parser.add_argument("-o", "--output", default="work/diarization.json")
    parser.add_argument("--num-speakers", type=int, default=None)
    args = parser.parse_args()

    diarizer = Diarizer()
    segments = diarizer.diarize(args.audio, num_speakers=args.num_speakers)
    save_diarization(segments, args.output)