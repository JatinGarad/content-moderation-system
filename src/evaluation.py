"""
evaluation.py
-------------
Evaluation utilities matching the metrics table in the project spec:

  Component               Metric                     Target
  ---------------------------------------------------------------
  ASR (Whisper)            Word Error Rate (WER)      < 10%
  Speaker Diarization      Diarization Error Rate     < 15%
  Profanity Detection      Precision / Recall / F1    baseline vs >90% F1
  End-to-End System        Detection Accuracy         > 85%

Requires reference (ground-truth) annotations to compare against --
you will need to hand-annotate a small test set of movie clips
(transcript, speaker turns, and profanity spans) to use this module.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Standard WER via Levenshtein distance at the word level:
    WER = (Substitutions + Deletions + Insertions) / len(reference words)
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

    return d[len(ref_words)][len(hyp_words)] / max(1, len(ref_words))


def diarization_error_rate(
    reference: List[Tuple[float, float, str]],
    hypothesis: List[Tuple[float, float, str]],
    resolution: float = 0.01,
) -> float:
    """
    Simplified frame-based Diarization Error Rate.

    Both reference and hypothesis are lists of (start, end, speaker_label)
    tuples. We discretise the timeline into frames of `resolution`
    seconds and compare speaker labels frame-by-frame after finding the
    optimal reference<->hypothesis label mapping (via majority overlap).

    Note: for rigorous DER (including confusion/missed-speech/false-alarm
    breakdown) use `pyannote.metrics.diarization.DiarizationErrorRate`
    directly on `pyannote.core.Annotation` objects -- this simplified
    version is for quick sanity checks during development.
    """
    if not reference or not hypothesis:
        return 1.0

    end_time = max(max(e for _, e, _ in reference), max(e for _, e, _ in hypothesis))
    n_frames = int(end_time / resolution) + 1

    def to_frame_labels(segments):
        labels = [None] * n_frames
        for start, end, spk in segments:
            for f in range(int(start / resolution), min(n_frames, int(end / resolution))):
                labels[f] = spk
        return labels

    ref_labels = to_frame_labels(reference)
    hyp_labels = to_frame_labels(hypothesis)

    # Majority-overlap mapping from hypothesis labels -> reference labels
    from collections import defaultdict
    overlap = defaultdict(lambda: defaultdict(int))
    for r, h in zip(ref_labels, hyp_labels):
        if r is not None and h is not None:
            overlap[h][r] += 1

    hyp_to_ref = {h: max(counts, key=counts.get) for h, counts in overlap.items()}

    errors = 0
    total = 0
    for r, h in zip(ref_labels, hyp_labels):
        if r is None and h is None:
            continue
        total += 1
        mapped_h = hyp_to_ref.get(h)
        if mapped_h != r:
            errors += 1

    return errors / max(1, total)


def precision_recall_f1(true_positives: int, false_positives: int, false_negatives: int):
    precision = true_positives / max(1, (true_positives + false_positives))
    recall = true_positives / max(1, (true_positives + false_negatives))
    f1 = 2 * precision * recall / max(1e-9, (precision + recall))
    return precision, recall, f1


def evaluate_against_spec_targets(results: dict):
    """
    Pretty-prints a pass/fail summary against the spec's target table.
    `results` should contain any subset of: wer, der, precision, recall,
    f1, end_to_end_accuracy, processing_time_ratio, sus_score.
    """
    targets = {
        "wer": ("ASR Word Error Rate", 0.10, "lower"),
        "der": ("Diarization Error Rate", 0.15, "lower"),
        "f1": ("Profanity Detection F1 (RoBERTa)", 0.90, "higher"),
        "end_to_end_accuracy": ("End-to-End Detection Accuracy", 0.85, "higher"),
        "processing_time_ratio": ("Processing Time (x real-time)", 2.0, "lower"),
        "sus_score": ("Usability (SUS Score)", 70, "higher"),
    }

    print(f"{'Metric':40s} {'Result':>10s} {'Target':>10s} {'Status':>8s}")
    print("-" * 72)
    for key, (label, target, direction) in targets.items():
        if key not in results:
            continue
        value = results[key]
        passed = (value <= target) if direction == "lower" else (value >= target)
        status = "PASS" if passed else "FAIL"
        print(f"{label:40s} {value:>10.3f} {target:>10.3f} {status:>8s}")
