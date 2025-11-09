# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The BESST Cognitive Load Dataset is a multimodal dataset for studying speech production under stress and cognitive load. It provides a Hugging Face datasets-compatible interface for loading speech data with various target types (cognitive load, physical load, speaker ID, gender) and modality combinations.

**Key characteristics:**
- 79 participants with 8 signal types
- Supports train/validation/test splits with 5 jackknifing variants (a-e)
- Targets: cognitive load (low/medium/high), physical load, speaker identification, gender
- Modality subsets: audio, audio-video, audio-video-bio, audio-video-ecg
- Raw data must be obtained separately from https://speech.fit.vut.cz/software/besst

## Architecture

The dataset package uses Hugging Face's `datasets` library with custom `GeneratorBasedBuilder` implementations:

### Dataset Variants

Three main dataset classes exist in `dataset/`:
1. **`dataset.py`** - Main cognitive/physical load classification dataset
2. **`speaker_dataset.py`** - Speaker identification (79 speakers)
3. **`gender_dataset.py`** - Gender classification (F/M)

All three share the same architecture but differ in their target labels:
- `dataset.py`: ClassLabel with 3 classes (low, medium, high)
- `speaker_dataset.py`: ClassLabel with 79 classes (speaker IDs 7-90)
- `gender_dataset.py`: ClassLabel with 2 classes (F, M)

### Configuration System

Each dataset class generates 40 configurations (BUILDER_CONFIGS) as combinations of:
- **target_type**: `cognitive-load` or `physical-load`
- **subset**: `audio`, `audio-video`, `audio-video-bio`, `audio-video-ecg`
- **split_variant**: `a`, `b`, `c`, `d`, `e` (jackknifing variants)

Configuration naming: `{target_type}_{subset}-{split_variant}`
Example: `cognitive-load_audio-video-a`

### Data Loading Pipeline

1. **Metadata**: `.scp` files in `dataset/metadata/lists/{target_type}/{subset}/{split_variant}/{train|validation|test}.scp`
2. **Speaker stats**: JSON files in `dataset/metadata/stats/` for speaker normalization (mean, std per participant)
3. **Raw data**: Must be provided via `data_dir` parameter
   - Audio: `{data_dir}/audio/{pid}/close_talk.wav`
   - Video: `{data_dir}/video/{segment_name}`
   - Bio/ECG: `{data_dir}/bio/` or `{data_dir}/ecg/`

### Audio Processing

The dataset performs efficient audio loading:
- Opens audio files once per participant and seeks to segment boundaries
- Applies speaker normalization: `(audio - mean) / std` using precomputed stats
- Resamples to target sampling rate (default 16kHz) using librosa
- Uses soundfile for efficient random access

## Loading the Dataset

Basic usage:
```python
from datasets import load_dataset

dataset = load_dataset(
    "path/to/dataset/dataset.py",  # or speaker_dataset.py, gender_dataset.py
    name="cognitive-load_audio-a",
    data_dir="/path/to/raw/data",
    metadata_dir="path/to/dataset/metadata",
    trust_remote_code=True,
    target_fs=16000,  # target sampling rate
    speaker_normalization=True  # apply per-speaker normalization
)
```

## Development Commands

### Installing Dependencies
```bash
pip install -r requirements.txt
# Core dependencies: numpy~=1.24.4, datasets~=2.20.0
# Also requires: librosa, soundfile
```

### Verification and Testing
```bash
# Verify metadata and raw data integrity
python utils/verify.py

# The verify script checks:
# - All metadata .scp files exist
# - All referenced audio files exist in data_dir
# - Dataset can be loaded successfully
```

### Generating Speaker Statistics
```bash
# Generate speaker normalization statistics (mean, std per speaker)
python utils/estimate_speaker_stats.py

# This creates JSON files in dataset/metadata/stats/
# Format: {pid: [mean, std, num_samples]}
# Required before loading datasets with speaker_normalization=True
```

## Metadata Structure

### File Organization
```
dataset/metadata/
├── participants.csv          # Participant demographics (pid, gender, etc.)
├── speech-segments.csv       # All speech segments with timing and labels
├── semantic-segments.csv     # Semantic annotations
├── lists/                    # Train/val/test splits
│   └── {target_type}/       # cognitive-load, physical-load
│       └── {subset}/        # audio, audio-video, etc.
│           └── {variant}/   # a, b, c, d, e
│               ├── train.scp
│               ├── validation.scp
│               └── test.scp
└── stats/                   # Precomputed speaker statistics
    └── {target_type}_{subset}-{variant}  # JSON files
```

### .scp File Format
Semicolon-delimited with 10 columns:
`pid;segid;start;end;field5;segment_name;field7;field8;field9;cognitive_load_label`

Key fields:
- `pid`: participant ID (e.g., "7", "8", etc.)
- `segid`: segment ID
- `start/end`: millisecond timestamps
- `segment_name`: video/bio file basename
- `cognitive_load_label`: "1" (low), "2" (medium), "3" (high)

## Important Implementation Details

### Dataset Differences
- **`dataset.py`**: skiprows=0 in np.loadtxt, uses cognitive_load_label for target
- **`speaker_dataset.py`**: skiprows=1, uses pid as target, 79-class output
- **`gender_dataset.py`**: skiprows=0, loads participants.csv for pid→gender mapping

### Memory Efficiency
- Keeps one SoundFile handle open per participant to avoid repeated I/O
- Seeks to segment boundaries rather than loading full audio files
- Uses `DEFAULT_WRITER_BATCH_SIZE = 50` for dataset generation

### Split Variants
The 5 jackknifing variants (a-e) provide different train/val/test partitions for cross-validation and robustness testing.

## Citation

When using this dataset, cite:
```bibtex
@article{Pesan2024,
  title = {Speech production under stress for machine learning: multimodal dataset of 79 cases and 8 signals},
  volume = {11},
  ISSN = {2052-4463},
  url = {http://dx.doi.org/10.1038/s41597-024-03991-w},
  DOI = {10.1038/s41597-024-03991-w},
  number = {1},
  journal = {Scientific Data},
  publisher = {Springer Science and Business Media LLC},
  author = {Pešán, Jan and Juřík, Vojtěch and Ružičková, Alexandra and others},
  year = {2024},
  month = nov
}
```
