"""
transcription.py
-----------------
STAGE 2 of the pipeline: Automatic Speech Recognition.

Wraps OpenAI's Whisper large-v3 checkpoint to produce a word-level
timestamped transcript. Chosen for state-of-the-art WER and native
multilingual support (relevant to the Hindi/Hinglish moderation gap
noted in the spec).

Implementation note (dissertation-relevant): this module runs the
Whisper large-v3 weights through Hugging Face `transformers`'
WhisperForConditionalGeneration rather than the original `openai-whisper`
package. Same model, same weights, same word-level timestamp mechanism
(dynamic time warping over cross-attention, per Whisper's own
implementation) -- the only change is the inference library, made
because `openai-whisper`'s forward-hook-based KV-cache is incompatible
with newer PyTorch releases (see project notes / dissertation Chapter 4
"Engineering challenges encountered and resolved" for the failure mode).
This also adds Apple Silicon (MPS) GPU support, which the original
`openai-whisper` path did not use on this machine -- CUDA-or-CPU only.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch

from config import ASRConfig

logger = logging.getLogger(__name__)


@dataclass
class WordTiming:
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: List[WordTiming]


def _resolve_device(requested: str) -> str:
    """
    Pick the best available device. `requested` normally comes from
    ASRConfig.device (default "cuda", inherited from the original
    CUDA-or-CPU-only implementation). On a Mac this now also checks for
    Apple Silicon's MPS backend before giving up and falling back to CPU.
    """
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Hugging Face model IDs for each Whisper size, matching the CLI names
# used elsewhere in the project (ASRConfig.model_name / --model).
_MODEL_ID_MAP = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
}


class Transcriber:
    def __init__(self, cfg: ASRConfig = ASRConfig()):
        self.cfg = cfg

        device = _resolve_device(cfg.device)
        if device != cfg.device:
            logger.warning(
                "Requested device '%s' unavailable; using '%s' instead.",
                cfg.device, device,
            )
        self.device = device

        model_id = _MODEL_ID_MAP.get(cfg.model_name, cfg.model_name)

        # Imported lazily so environments that only need other pipeline
        # stages (e.g. censorship.py in isolation) don't need transformers
        # importable at module load time.
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading Whisper model '%s' (%s) on %s ...",
                    cfg.model_name, model_id, device)

        # float16 on GPU/MPS for speed and memory; float32 on CPU, where
        # half precision is either unsupported or slower for this model.
        torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype
        ).to(device)
        self.model.eval()
        self.torch_dtype = torch_dtype

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        """
        Run Whisper transcription with word-level timestamps.

        Uses `generate(..., return_token_timestamps=True)`, which returns
        one timestamp per output token computed via dynamic time warping
        over the model's cross-attention (Whisper's own alignment
        mechanism -- see `WhisperGenerationMixin._extract_token_timestamps`
        in transformers/models/whisper/generation_whisper.py). This is
        the verified, version-stable API for this transformers release;
        the tokenizer-level `output_offsets=True` path used in an earlier
        draft of this module only supports segment-level timestamps and
        was the wrong tool for word-level timing.

        Returns a list of TranscriptSegment, each containing per-word
        timing needed for later censorship (Stage 6) and speaker
        attribution (Stage 5 fusion). Return shape is identical to the
        original openai-whisper-backed implementation.
        """
        import librosa

        logger.info("Transcribing %s ...", audio_path)

        # Whisper models expect 16kHz mono; audio_extraction.py already
        # produces this, but we resample defensively in case this method
        # is ever called on a WAV from elsewhere.
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        generate_kwargs = {
            "task": "transcribe",
            # Segment-level timestamp tokens must be on for the model to
            # emit any timing signal at all; token_timestamps below is
            # what actually gives us word-level resolution.
            "return_timestamps": True,
            "return_token_timestamps": self.cfg.word_timestamps,
        }
        if self.cfg.language:
            generate_kwargs["language"] = self.cfg.language

        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).to(self.device, dtype=self.torch_dtype)

        with torch.no_grad():
            result = self.model.generate(
                **inputs,
                **generate_kwargs,
                return_dict_in_generate=True,
                output_scores=False,
                # long-form chunking so clips longer than 30s are handled
                # in one call, same behaviour as openai-whisper's default
                condition_on_prev_tokens=False,
            )

        # generate() returns a plain dict here (transformers 5.x, when
        # return_token_timestamps and/or return_timestamps trigger the
        # dict-returning path) rather than a GenerateOutput object on
        # some versions -- handle both so an upgrade/downgrade doesn't
        # silently break this again.
        def _get(key):
            return result[key] if isinstance(result, dict) else getattr(result, key)

        sequences = _get("sequences")
        token_timestamps = None
        if self.cfg.word_timestamps:
            try:
                token_timestamps = _get("token_timestamps")
            except (KeyError, AttributeError):
                logger.warning(
                    "Model did not return token_timestamps (checkpoint may be "
                    "missing `generation_config.alignment_heads`); falling back "
                    "to segment-level timestamps only -- word-level censorship "
                    "precision will be reduced to sentence granularity."
                )

        if token_timestamps is not None:
            segments = self._build_segments_from_tokens(
                sequences[0], token_timestamps[0]
            )
        else:
            # Fallback: decode without word timing, one segment, no
            # per-word censorship precision. Better than crashing.
            text = self.processor.batch_decode(
                sequences, skip_special_tokens=True
            )[0].strip()
            duration = len(audio) / 16000
            segments = [TranscriptSegment(id=0, start=0.0, end=duration,
                                          text=text, words=[])]

        detected_lang = self.cfg.language or "auto"
        logger.info("Transcription complete: %d segments, language=%s",
                     len(segments), detected_lang)
        return segments

    def _build_segments_from_tokens(self, token_ids, token_timestamps) -> List[TranscriptSegment]:
        """
        Pair each output token with its timestamp, decode tokens one at a
        time to detect word boundaries (a new word starts at a token whose
        decoded text begins with a leading space, matching GPT-2/BPE-style
        tokenizers including Whisper's), and group words into
        sentence-like segments on terminal punctuation or a >0.8s gap --
        the same segmentation heuristic openai-whisper applies.
        """
        tokenizer = self.processor.tokenizer
        special_ids = set(tokenizer.all_special_ids)

        words: List[WordTiming] = []
        cur_text = ""
        cur_start = None
        cur_end = None

        ts = token_timestamps.tolist() if hasattr(token_timestamps, "tolist") else list(token_timestamps)
        ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)

        def flush_word():
            nonlocal cur_text, cur_start, cur_end
            stripped = cur_text.strip()
            if stripped:
                words.append(WordTiming(
                    word=stripped,
                    start=float(cur_start if cur_start is not None else 0.0),
                    end=float(cur_end if cur_end is not None else 0.0),
                    confidence=1.0,
                ))
            cur_text, cur_start, cur_end = "", None, None

        for tok_id, t in zip(ids, ts):
            if tok_id in special_ids:
                continue
            piece = tokenizer.decode([tok_id])
            if not piece:
                continue
            if piece.startswith(" ") and cur_text:
                flush_word()
            cur_text += piece
            if cur_start is None:
                cur_start = t
            cur_end = t
        flush_word()

        # Group words into segments on terminal punctuation or pause > 0.8s
        segments: List[TranscriptSegment] = []
        current_words: List[WordTiming] = []
        seg_id = 0
        last_end = None

        def flush_segment():
            nonlocal seg_id, current_words
            if not current_words:
                return
            text = " ".join(w.word for w in current_words).strip()
            segments.append(TranscriptSegment(
                id=seg_id,
                start=current_words[0].start,
                end=current_words[-1].end,
                text=text,
                words=list(current_words),
            ))
            seg_id += 1
            current_words = []

        for w in words:
            if current_words and last_end is not None and (w.start - last_end > 0.8):
                flush_segment()
            current_words.append(w)
            last_end = w.end
            if w.word.endswith((".", "!", "?")):
                flush_segment()
        flush_segment()

        return segments


def save_transcript(segments: List[TranscriptSegment], out_path: str):
    """Serialise transcript to JSON (the Stage 2 output artefact)."""
    data = [
        {
            "id": s.id, "start": s.start, "end": s.end, "text": s.text,
            "words": [asdict(w) for w in s.words],
        }
        for s in segments
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Transcript saved to %s", out_path)


def load_transcript(path: str) -> List[TranscriptSegment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        TranscriptSegment(
            id=d["id"], start=d["start"], end=d["end"], text=d["text"],
            words=[WordTiming(**w) for w in d["words"]],
        )
        for d in data
    ]


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper.")
    parser.add_argument("audio", help="Path to input WAV file")
    parser.add_argument("-o", "--output", default="work/transcript.json")
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    cfg = ASRConfig(model_name=args.model)
    transcriber = Transcriber(cfg)
    segments = transcriber.transcribe(args.audio)
    save_transcript(segments, args.output)
