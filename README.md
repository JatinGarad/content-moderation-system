# Multimodal Profanity Detection and Censorship for Film Audio

An end-to-end pipeline that finds offensive speech in a video, works out who
said it, and produces a censored version. It combines Whisper speech
recognition, pyannote speaker diarization and a lexical/transformer detection
stage, then reconciles their outputs on a single timeline so that every flagged
word carries its exact timing and its speaker.

I built this as my MSc project (COMP702) at the University of Liverpool.

## How it works

| Stage | Module | Output |
|---|---|---|
| 1. Audio extraction | `src/audio_extraction.py` | 16 kHz mono WAV, loudness-normalised |
| 2. Speech recognition | `src/transcription.py` | transcript with a timestamp for every word (Whisper large-v3) |
| 3. Speaker diarization | `src/diarization.py` | speaker turns (pyannote 3.1) |
| 4. Offensive language detection | `src/profanity_detection.py` | flagged words, from a lexical path and a RoBERTa path |
| 5. Fusion | `src/fusion.py` | one record per flagged word: word, interval, speaker, severity |
| 6. Censorship output | `src/censorship.py` | censored video, masked `.srt` subtitles, PDF report |

The stages only talk to each other through JSON files on disk, so any stage
can be run, inspected and re-run on its own. That separation is what let me
swap out the whole speech recognition backend partway through the project
without changing anything downstream.

Fusion is the interesting part. Whisper's word boundaries and pyannote's
speaker boundaries come from two different models and never line up, so
`fusion.py` resolves each flagged word at its temporal midpoint against the
speaker turns. Because the censored audio, the subtitles and the report are
all generated from that one fused record, they can't disagree about what was
flagged.

## Current status

The full pipeline runs end to end without manual intervention. On a
16.84-second test clip it averaged a real-time factor of 1.54 (range
1.45–1.63 over three runs). About 58% of that time is speech recognition and
28% is diarization, so speeding up longer inputs is mainly a question of
optimising the ASR stage.

It is a research prototype, and these are the known issues:

- **Ensemble mode never uses RoBERTa.** In `ensemble` mode the combination
  rule reduces to the lexical result, so the RoBERTa branch is unreachable
  and every detection currently comes from the word list.
- **No fine-tuned classifier is included.** `train_roberta.py` implements
  fine-tuning on the Jigsaw corpus but has not been run, so the RoBERTa path
  falls back to `roberta-base` with an untrained classification head.
- **Subtitle masking misses words with punctuation attached.** On the test
  clip `bullshit.` is left unmasked in the `.srt`, while the audio for the
  same word is still beeped correctly.
- **Single-token words get zero-length intervals**, so their beep covers only
  the padding either side.
- **Speaker attribution is strict.** A word whose midpoint falls in a gap
  between speaker turns comes back as `UNKNOWN`, and overlapping speech is
  not reconciled.
- **Accuracy has not been measured.** Transcription, diarization and
  detection accuracy all need a hand-annotated corpus of film dialogue, which
  doesn't exist yet. This is the next piece of work.

## Setup

You need Python and FFmpeg.

```bash
# FFmpeg is not installable with pip
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Ubuntu / Debian

pip install -r requirements.txt
```

If `openai-whisper` fails to install, the comments at the bottom of
`requirements.txt` explain the fix.

Diarization uses a gated model. Create a Hugging Face access token, accept the
terms at https://huggingface.co/pyannote/speaker-diarization-3.1, then either
set the token:

```bash
export HF_TOKEN=<your-huggingface-token>
```

or log in once with `huggingface-cli login`. The code uses whichever it finds.

## Usage

Run the whole pipeline on a video:

```bash
cd src
python main.py /path/to/video.mp4 --outdir ../output
```

This writes `<name>_censored.mp4`, `<name>_clean.srt` and `<name>_report.pdf`
to `output/`. `--profanity-method` takes `rule_based`, `roberta` or
`ensemble` (the default), and `--num-speakers` fixes the speaker count if you
know it.

Or use the interface:

```bash
cd src
streamlit run dashboard.py   # four pages: process, results, compare methods, evaluation
streamlit run app.py         # minimal single-page version
```

Each stage can also be run on its own, which is useful for debugging:

```bash
cd src
python audio_extraction.py video.mp4 -o work/audio.wav
python transcription.py work/audio.wav -o work/transcript.json --model large-v3
python diarization.py work/audio.wav -o work/diarization.json
python profanity_detection.py work/transcript.json -o work/flagged.json --method ensemble
python fusion.py --transcript work/transcript.json --diarization work/diarization.json \
                 --flagged work/flagged.json -o work/timeline.json
python censorship.py --audio work/audio.wav --video video.mp4 \
                     --transcript work/transcript.json --timeline work/timeline.json \
                     --outdir output
```

`examples/` holds the JSON artefacts from the test clip, so you can try the
later stages without running the models.

### Fine-tuning the classifier (optional)

Download `train.csv` from the
[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge),
then:

```bash
cd src
python train_roberta.py --data path/to/train.csv
```

The checkpoint is saved to `src/models/roberta-profanity`, which is where the
detection stage looks for it. Note that the ensemble issue above still has to
be fixed before a trained model changes the ensemble output.

### Evaluation

`src/evaluation.py` provides `word_error_rate`, `diarization_error_rate`,
`precision_recall_f1` and `evaluate_against_spec_targets`. They all need
ground truth: a reference transcript, reference speaker turns and marked
profanity spans.

Two caveats. The DER target of 0.15 in `evaluate_against_spec_targets` comes
from my original specification, and it turned out to be a broadcast-news
figure; the film and television benchmark for the same pyannote system is
roughly six times higher, so 0.40–0.50 is a realistic target for film audio.
And `diarization_error_rate` here is a simplified frame-based version; for
numbers comparable with published work, use
`pyannote.metrics.diarization.DiarizationErrorRate`.

## Built with

[Whisper](https://github.com/openai/whisper) (run through Hugging Face
`transformers`), [pyannote.audio](https://github.com/pyannote/pyannote-audio),
[RoBERTa](https://huggingface.co/roberta-base),
[better-profanity](https://github.com/snguyenthanh/better_profanity),
FFmpeg and Streamlit.
