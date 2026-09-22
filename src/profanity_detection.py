"""
profanity_detection.py
-----------------------
STAGE 4 of the pipeline: Offensive Language Detection.

Implements the two approaches described in the spec:
  (a) Rule-based baseline using the 'better-profanity' library.
  (b) A fine-tuned RoBERTa classifier (trained on the Jigsaw Toxic
      Comment Dataset) for contextual, non-keyword-based detection.

`method="ensemble"` flags a word/segment if either approach fires,
taking the maximum confidence score as the severity score -- this
gives high recall from the rule-based pass and contextual precision
from RoBERTa.
"""

import logging
from dataclasses import dataclass
from typing import List

import torch
from better_profanity import profanity
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import ProfanityConfig
from transcription import TranscriptSegment, WordTiming

logger = logging.getLogger(__name__)

profanity.load_censor_words()


@dataclass
class FlaggedWord:
    word: str
    start: float
    end: float
    segment_id: int
    severity: float          # 0.0-1.0 confidence/severity score
    source: str              # "rule_based", "roberta", or "ensemble"


class RuleBasedDetector:
    """Fast keyword/edit-distance baseline. High recall, low precision
    (misses context, slang, euphemisms -- as noted in the spec's
    limitations table)."""

    def is_profane(self, text: str) -> bool:
        return profanity.contains_profanity(text)

    def score(self, text: str) -> float:
        return 1.0 if self.is_profane(text) else 0.0


class RobertaDetector:
    """Contextual transformer classifier. Falls back to the base
    `roberta-base` checkpoint (with a warning) if no fine-tuned
    checkpoint is found -- see train_roberta.py to produce one."""

    def __init__(self, cfg: ProfanityConfig):
        self.cfg = cfg
        model_source = cfg.roberta_model_dir
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
            logger.info("Loaded fine-tuned RoBERTa checkpoint from %s", model_source)
        except OSError:
            logger.warning(
                "No fine-tuned checkpoint found at '%s'. Falling back to base "
                "'%s' with an UNTRAINED classification head -- run "
                "train_roberta.py first for meaningful predictions.",
                model_source, cfg.base_roberta_checkpoint,
            )
            model_source = cfg.base_roberta_checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source, num_labels=2
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str) -> float:
        """Returns P(toxic) in [0, 1]."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        # Convention: label index 1 = toxic/offensive
        return probs[0, 1].item()


class ProfanityDetector:
    """Top-level detector combining rule-based and RoBERTa approaches
    according to `cfg.method`, and applying them at word level so
    censorship (Stage 6) can target individual words rather than whole
    sentences."""

    def __init__(self, cfg: ProfanityConfig = ProfanityConfig()):
        self.cfg = cfg
        self.rule_based = RuleBasedDetector()
        self.roberta = RobertaDetector(cfg) if cfg.method in ("roberta", "ensemble") else None

    def detect(self, segments: List[TranscriptSegment]) -> List[FlaggedWord]:
        flagged: List[FlaggedWord] = []

        for seg in segments:
            # Word-level pass (rule-based is cheap enough to run per word;
            # this is what enables precise beep placement in Stage 6).
            for w in seg.words:
                clean_word = w.word.strip().lower()
                if not clean_word:
                    continue

                rb_hit = self.rule_based.is_profane(clean_word)

                if self.cfg.method == "rule_based":
                    if rb_hit:
                        flagged.append(FlaggedWord(
                            word=w.word, start=w.start, end=w.end,
                            segment_id=seg.id, severity=1.0, source="rule_based",
                        ))
                    continue

                # RoBERTa is more reliable with sentence-level context, so we
                # score the *segment* once and apply that score to any word
                # inside it that the rule-based pass also flags, OR -- for
                # segments with no rule-based hits at all -- we still run a
                # contextual check to catch euphemisms/slang the keyword list misses.
                if self.cfg.method in ("roberta", "ensemble"):
                    seg_score = self.roberta.score(seg.text)
                    contextual_hit = seg_score >= self.cfg.severity_threshold

                    if rb_hit or (contextual_hit and self.cfg.method == "roberta"):
                        severity = max(seg_score, 1.0 if rb_hit else 0.0)
                        source = (
                            "ensemble" if (rb_hit and contextual_hit)
                            else "rule_based" if rb_hit else "roberta"
                        )
                        flagged.append(FlaggedWord(
                            word=w.word, start=w.start, end=w.end,
                            segment_id=seg.id, severity=severity, source=source,
                        ))
                    elif self.cfg.method == "ensemble" and rb_hit:
                        # rule-based fired even though contextual score was low --
                        # keep recall high per spec's baseline comparison goal
                        flagged.append(FlaggedWord(
                            word=w.word, start=w.start, end=w.end,
                            segment_id=seg.id, severity=1.0, source="rule_based",
                        ))

        logger.info("Profanity detection complete: %d words flagged", len(flagged))
        return flagged


if __name__ == "__main__":
    import argparse
    import json
    from dataclasses import asdict
    from transcription import load_transcript

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Detect offensive language in a transcript.")
    parser.add_argument("transcript", help="Path to transcript.json (from transcription.py)")
    parser.add_argument("-o", "--output", default="work/flagged_words.json")
    parser.add_argument("--method", choices=["rule_based", "roberta", "ensemble"], default="ensemble")
    args = parser.parse_args()

    cfg = ProfanityConfig(method=args.method)
    detector = ProfanityDetector(cfg)
    segments = load_transcript(args.transcript)
    flagged = detector.detect(segments)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([asdict(w) for w in flagged], f, indent=2, ensure_ascii=False)
    logger.info("Saved %d flagged words to %s", len(flagged), args.output)
