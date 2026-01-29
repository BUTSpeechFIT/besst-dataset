# BESST Cognitive Load Dataset

The **BESST Cognitive Load Dataset** is a multimodal dataset designed to study speech production under stress and cognitive load. It contains empirical data collected using the BESST experimental protocol, featuring **79 cases** and **8 signals**. This dataset is aimed at advancing machine learning models for stress estimation and pre-emptive intervention systems.

## Table of Contents
1. [Overview](#overview)
2. [Dataset Features](#dataset-features)
3. [Experimental Protocol](#experimental-protocol)
4. [Getting Started](#getting-started)
5. [Usage](#usage)
6. [Citation](#citation)
7. [License](#license)

---

## Overview

Stress can impair cognitive functions, impacting critical areas like aviation, surgery, and public safety. The BESST Cognitive Load Dataset provides multimodal data for developing advanced machine learning models to identify stress and cognitive load from speech. This dataset bridges the gap in available resources, offering high-quality data annotated with subjective self-determined and objective biological labels.

---

## Dataset Features

| Field            | Type           | Description                                     |
|------------------|----------------|-------------------------------------------------|
| `id`             | `string`       | Unique identifier for each sample.              |
| `audio`          | `Audio`        | Audio array dict with `array` (numpy) and `sampling_rate`. |
| `pid`            | `int32`        | Participant ID.                                 |
| `start`          | `int32`        | Start time of the segment (in milliseconds).    |
| `end`            | `int32`        | End time of the segment (in milliseconds).      |
| `cognitive_load` | `ClassLabel`   | Cognitive load level (`low`, `medium`, `high`). |

---

## Experimental Protocol

The BESST protocol was used to collect data under controlled conditions. Participants performed tasks designed to induce cognitive load and stress while speech, physiological, and behavioral signals were recorded.

### Signals Captured:
1. Speech signals (close-talk microphones)
2. ECG+Heart rate and variability
3. Skin conductance
4. Task performance metrics
5. Reaction times
6. Other relevant behavioral data

---

## Getting Started

### Prerequisites
- Python 3.8 or later
- Hugging Face `datasets` library

### Installation

**Option 1: Editable install (for development)**
```bash
git clone https://github.com/BUTSpeechFIT/besst-dataset.git
cd besst-dataset
pip install -e .
```

**Option 2: Direct install from git**
```bash
pip install git+https://github.com/BUTSpeechFIT/besst-dataset.git@v1.0.0
```

### Usage
Loading the Dataset
Load the dataset using Hugging Face's datasets library:

```python
from datasets import load_dataset

# Load cognitive-load dataset (v1.0: direct audio loading with normalization)
# Audio-only, split variant 'a'
dataset = load_dataset(
    "dataset/dataset.py",  # After installation
    name="cognitive-load_audio-a",
    data_dir="/path/to/raw/data",  # Path to audio files
    trust_remote_code=True,
    target_fs=16000,              # Target sampling rate
    speaker_normalization=True,   # Normalize per speaker
)

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]
```

### Dataset Variants

The package provides three dataset classes for different classification tasks:

1. **`dataset/dataset.py`** - Cognitive/physical load classification
   - 3 classes: `low`, `medium`, `high`
   - Configurations: `cognitive-load_audio-{a,b,c,d,e}`, `physical-load_audio-{a,b,c,d,e}`

2. **`dataset/speaker_dataset.py`** - Speaker identification
   - 79 classes: speaker IDs 7-90
   - Configurations: `cognitive-load_audio-{a,b,c,d,e}`, `physical-load_audio-{a,b,c,d,e}`

3. **`dataset/gender_dataset.py`** - Gender classification
   - 2 classes: `F` (female), `M` (male)
   - Configurations: `cognitive-load_audio-{a,b,c,d,e}`, `physical-load_audio-{a,b,c,d,e}`

All variants support the same parameters (`target_fs`, `speaker_normalization`) and provide identical audio processing.

### Version History

**v1.0.0** (2026-01-29)
- **Major**: Proper jack-knifing methodology - ONE shared test set across all 5 folds (a-e), enabling valid averaging of test results
- Direct audio loading (returns numpy arrays instead of file paths)
- Speaker normalization with precomputed stats (`speaker_normalization=True`)
- Audio resampling to target sample rate (`target_fs=16000`)
- New `pid` field in output (participant ID)
- New dataset variants: speaker identification (`speaker_dataset.py`), gender classification (`gender_dataset.py`)
- Simplified cognitive-load lists to audio-only (multi-modal subsets removed)

**v0.6.0** (2025-11-09)
- Added BESST v6 with phase-based cognitive load splits
- Fixed jack-knifing methodology (equalized test sets across folds)
- Documented class imbalance (compensated via loss weighting)
- Excluded learning phases (R0) and debriefing
- All 71 participants in train/val/test (within-subject design)

**v0.5.0** (2024)
- Initial release with participant-based splits

### Citation
If you use this dataset in your research, please cite it as follows:

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
          author = {Pešán, Jan and Juřík, Vojtěch and Ružičková, Alexandra and Svoboda, Vojtěch and Janoušek, Oto and Němcová, Andrea and Bojanovská, Hana and Aldabaghová, Jasmína and Kyslík, Filip and Vodičková, Kateřina and Sodomová, Adéla and Bartys, Patrik and Chudý, Peter and Černocký, Jan},
          year = {2024},
          month = nov
        }
```
### License

See: [speech.fit.vut.cz/software/besst](speech.fit.vut.cz/software/besst)
