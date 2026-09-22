"""
config.py
---------
Central configuration for the content moderation pipeline.
Values here map directly onto the six-stage pipeline described in the
project spec (Audio Extraction -> ASR -> Diarization -> Offensive
Language Detection -> Multimodal Fusion -> Content Censorship).
"""

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class AudioConfig:
    sample_rate: int = 16000          # Whisper/PyAnnote expect 16kHz mono
    channels: int = 1
    format: str = "wav"


@dataclass
class ASRConfig:
    model_name: str = "large-v3"      # OpenAI Whisper model size
    device: str = "cuda"              # falls back to "cpu" automatically if unavailable
    language: str = None              # None = auto-detect (supports multilingual, e.g. Hindi/Hinglish)
    word_timestamps: bool = True


@dataclass
class DiarizationConfig:
    model_name: str = "pyannote/speaker-diarization-3.1"
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))
    device: str = "cuda"


@dataclass
class ProfanityConfig:
    # "rule_based" -> better-profanity baseline
    # "roberta"    -> fine-tuned transformer classifier
    # "ensemble"   -> flag if either fires, score = max(confidence)
    method: str = "ensemble"
    roberta_model_dir: str = "models/roberta-profanity"  # local fine-tuned checkpoint
    base_roberta_checkpoint: str = "roberta-base"          # used if no fine-tuned model exists yet
    severity_threshold: float = 0.5   # classifier confidence above which a word/segment is censored


@dataclass
class CensorshipConfig:
    beep_freq_hz: int = 1000
    beep_gain_db: float = -3.0        # beep loudness relative to 0 dBFS
    padding_ms: int = 40              # small buffer added before/after each censored span


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    profanity: ProfanityConfig = field(default_factory=ProfanityConfig)
    censorship: CensorshipConfig = field(default_factory=CensorshipConfig)

    work_dir: Path = Path("work")
    output_dir: Path = Path("output")

    def ensure_dirs(self):
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = PipelineConfig()
