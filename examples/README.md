# Example artefacts

These are the JSON files the pipeline wrote for the 16.84-second test clip
used in the evaluation, copied unchanged. The clip itself is not included.

| File | Written by | Contents |
|---|---|---|
| `transcript.json` | Stage 2 | 8 segments, 47 words, each with its own start and end time |
| `diarization.json` | Stage 3 | speaker turns (one speaker on this clip) |
| `timeline.json` | Stage 5 | the two flagged words, each with its interval, speaker and severity |
| `run_history.json` | pipeline | per-stage timings for three runs, used for the real-time factor |

They are useful for trying the later stages without running the models. For
example, from the repository root:

```bash
cd src
python fusion.py --transcript ../examples/transcript.json \
                 --diarization ../examples/diarization.json \
                 --flagged flagged.json -o timeline.json
```

where `flagged.json` comes from
`python profanity_detection.py ../examples/transcript.json -o flagged.json --method rule_based`
(the `rule_based` path needs no model download).
