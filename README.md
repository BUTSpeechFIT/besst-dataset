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

| Field            | Type           | Description                                                                 |
|------------------|----------------|-----------------------------------------------------------------------------|
| `id`             | `string`       | Unique identifier for each sample.                                         |
| `audio`          | `Audio`        | Path to the audio file (supports resampling).                              |
| `start`          | `int32`        | Start time of the segment (in seconds).                                    |
| `end`            | `int32`        | End time of the segment (in seconds).                                      |
| `cognitive_load` | `ClassLabel`   | Cognitive load level (`low`, `medium`, `high`).                            |

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
- Python 3.7 or later
- Hugging Face `datasets` library: `pip install datasets`

### Installation
Clone the repository:
```bash
git clone https://github.com/BUTSpeechFIT/besst-dataset.git
cd dataset-dataset
```

### Usage
Loading the Dataset
Load the dataset using Hugging Face's datasets library:

```python
from datasets import load_dataset
# Load cognitive-load dataset, audio-video subset, split variant 'a'
dataset = load_dataset(
    "besst_dataset",
    name="cognitive-load_audio-video_a",
    data_dir="/path/to/data",
)

```

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
          author = {Pešán,  Jan and Juřík,  Vojtěch and Ružičková,  Alexandra and Svoboda,  Vojtěch and Janoušek,  Oto and Němcová,  Andrea and Bojanovská,  Hana and Aldabaghová,  Jasmína and Kyslík,  Filip and Vodičková,  Kateřina and Sodomová,  Adéla and Bartys,  Patrik and Chudý,  Peter and Černocký,  Jan},
          year = {2024},
          month = nov
        }
```
### License

See: [speech.fit.vut.cz/software/besst](speech.fit.vut.cz/software/besst)
