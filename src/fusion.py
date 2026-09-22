"""
fusion.py
---------
STAGE 5 of the pipeline: Multimodal Fusion.

Aligns the ASR transcript, speaker diarization, and offensive-word
flags on a shared timeline, producing a single attributed
offensive-word timeline: who said what, and when.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from transcription import TranscriptSegment
from diarization import SpeakerSegment, speaker_at
from profanity_detection import FlaggedWord

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    word: str
    start: float
    end: float
    speaker: str
    severity: float
    source: str
    segment_text: str


def fuse(
    segments: List[TranscriptSegment],
    diarization: List[SpeakerSegment],
    flagged_words: List[FlaggedWord],
) -> List[TimelineEvent]:
    """
    Build the attributed offensive-word timeline.

    For each flagged word, look up which speaker was active at its
    midpoint timestamp, and attach the full segment text for report
    context.
    """
    seg_by_id = {s.id: s for s in segments}
    timeline: List[TimelineEvent] = []

    for fw in flagged_words:
        midpoint = (fw.start + fw.end) / 2
        speaker = speaker_at(diarization, midpoint)
        seg = seg_by_id.get(fw.segment_id)
        segment_text = seg.text if seg else ""

        timeline.append(TimelineEvent(
            word=fw.word,
            start=fw.start,
            end=fw.end,
            speaker=speaker,
            severity=fw.severity,
            source=fw.source,
            segment_text=segment_text,
        ))

    timeline.sort(key=lambda e: e.start)
    logger.info("Fusion complete: %d attributed offensive-word events across %d speakers",
                len(timeline), len({e.speaker for e in timeline}))
    return timeline


def save_timeline(timeline: List[TimelineEvent], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in timeline], f, indent=2, ensure_ascii=False)
    logger.info("Timeline saved to %s", out_path)


def load_timeline(path: str) -> List[TimelineEvent]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [TimelineEvent(**d) for d in data]


if __name__ == "__main__":
    import argparse
    from transcription import load_transcript
    from diarization import load_diarization
    import dataclasses

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Fuse transcript, diarization, and profanity flags.")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--diarization", required=True)
    parser.add_argument("--flagged", required=True)
    parser.add_argument("-o", "--output", default="work/timeline.json")
    args = parser.parse_args()

    segments = load_transcript(args.transcript)
    diar = load_diarization(args.diarization)
    with open(args.flagged) as f:
        flagged_raw = json.load(f)
    flagged = [FlaggedWord(**d) for d in flagged_raw]

    timeline = fuse(segments, diar, flagged)
    save_timeline(timeline, args.output)
