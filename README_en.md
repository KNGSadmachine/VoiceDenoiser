# VoiceDenoiser

[日本語](README.md)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

Automated editing that protects the localization, distance, and left/right differences of binaural audio.

**Phase 1: De-plosive → Mouth De-click** is now implemented. It detects and locally repairs plosives and short mouth clicks in raw two-channel recordings, then produces a 32-bit float WAV intermediate master and a processing report for manual repair.

```text
Source
  ↓
Phase 1: De-plosive → Mouth De-click  ← available
  ↓
Manual repair
  ↓
Phase 2: Noise Reduction → EQ → DeEsser
  ↓
Manual level automation → Loudness matching
```

See [roadmap.md](roadmap.md) for the planned implementation order and quality gates.

## Phase 1 features

- **Binaural 2-channel only** — never mixes L/R; a one-sided event is repaired only in that channel
- **Fixed order** — De-plosive always runs before Mouth De-click
- **Localized processing** — plosives receive event-local low-frequency attenuation; clicks receive short interpolation without changing sample count
- **Conservative automation** — only strong candidates are repaired; ambiguous candidates are left unchanged and reported for review
- **Spatial and timing invariants** — preserves sample rate, two channels, sample count, and L/R order
- **Intermediate master** — writes 32-bit float WAV to avoid repeated quantization
- **Processing report** — JSON includes event time, channel, confidence, correction, and before/after peak and RMS
- **Safe resume** — skips output only when source content, settings, and processing version match
- **Atomic output** — an interrupted write is never treated as a finished file
- **Preview and batch processing** — compare one file, then process a directory while preserving its structure

Phase 1 does not normalize, trim silence, reduce steady noise, EQ, or de-ess.

## Requirements

- Python 3.10 or 3.11
- Windows / Linux / macOS
- Phase 1 runs on CPU
- An NVIDIA GPU is recommended only for Legacy denoising

## Setup

```bash
git clone https://github.com/ReineHonoka/VoiceDenoiser.git
cd VoiceDenoiser
setup.bat        # Windows
# ./setup.sh     # Linux / macOS
```

The setup creates a venv and installs PyTorch, dependencies, and the AI models used by Legacy denoising.

Use `SKIP_MODEL_DOWNLOAD=1 ./setup.sh` to postpone model downloads. Phase 1 itself does not use an AI model.

## Usage

```bash
run.bat          # Windows
# ./run.sh       # Linux / macOS
```

Your browser opens `http://127.0.0.1:7860`.

1. Put two-channel audio in `dataset/raw/`, drop files into the GUI, or choose another input folder
2. Keep **Phase 1 — De-plosive → Mouth De-click** selected
3. Convert one preview and compare before/after
4. Start the batch
5. Send the WAV files in `dataset/phase1/` to manual repair

The directory structure and base file names are preserved. Phase 1 output always uses the `.wav` extension.

### Reports

Each output gets a `.phase1.json` report beside it:

```text
dataset/phase1/scene/take.wav
dataset/phase1/scene/take.phase1.json
```

Summary fields:

- `de_plosive`: automatically repaired plosives
- `mouth_de_click`: automatically repaired mouth clicks
- `review`: candidates left unchanged for human review

Each event records L/R, start and end times, confidence, and the action taken.

## Input contract

- Exactly two channels
- At least 16 kHz
- At least 50 ms
- wav / flac / mp3 / ogg

Phase 1 does not silently convert mono or multichannel input because that could change spatial intent. An incompatible file fails individually while the rest of a batch continues.

## Legacy denoising

The previous DeepFilterNet / Resemble Enhance workflow remains available as **Legacy — 従来のノイズ除去**. Expand **Legacyノイズ除去の設定** to choose its engine, strength, processing mode, and post-processing.

Legacy is separate from Phase 1 and is never applied automatically to Phase 1 output.

## Tests

```bash
./venv/bin/python -m unittest discover -s tests -v
```

Tests cover synthetic plosive/click repair, fixed step order, channel and sample-count invariants, untouched opposite channels, 32-bit float WAV output, sidecars, and resume identity.

## License

[MIT License](LICENSE)

## Credits

- Legacy denoising model: [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (MIT / Apache-2.0)
- Legacy denoising model: [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (MIT) — inference code bundled in `resemble_enhance/`
- UI theme: [NoCrypt/miku](https://huggingface.co/spaces/NoCrypt/miku) (Apache-2.0)
