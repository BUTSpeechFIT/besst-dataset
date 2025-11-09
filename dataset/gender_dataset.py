import json
import logging
import os
import shutil
from argparse import ArgumentParser, Namespace

import datasets
import librosa
import numpy as np
import soundfile as sf

load2label = {"1": "low", "2": "medium", "3": "high"}
logging.basicConfig(level=logging.INFO)

class BESSTDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for BESST dataset."""

    def __init__(self, target_type, subset, split_variant, **kwargs):
        """
        Args:
            target_type (str): The target type ('cognitive-load' or 'physical-load').
            subset (str): The subset ('audio', 'audio-video', etc.).
            split_variant (str): The jackknifing split variant ('a', 'b', 'c', 'd', 'e').
            **kwargs: Additional keyword arguments passed to BuilderConfig.
        """
        super().__init__(**kwargs)
        self.target_type = target_type
        self.subset = subset
        self.split_variant = split_variant
        self.metadata_dir = kwargs.get("metadata_dir")


class BESSTDataset(datasets.GeneratorBasedBuilder):
    def __init__(self, *args, target_fs=16000, speaker_normalization=True, **kwargs):
        self.logger = datasets.logging.get_logger(f"BESSTSpeakerDataset")

        self.target_fs = target_fs
        self.speaker_normalization = speaker_normalization
        self.logger.info(f"Loading BESST dataset. fs: {target_fs} Hz, speaker_normalization: {speaker_normalization}")
        super().__init__(*args, **kwargs)


    BUILDER_CONFIGS = [
        BESSTDatasetConfig(
            name=f"{target_type}_{subset}-{split}",
            target_type=target_type,
            subset=subset,
            split_variant=split,
            description=f"{target_type} dataset with {subset} modalities and split variant {split}.",
            version=datasets.Version("0.1.0"),
        )
        for target_type in ["cognitive-load", "physical-load"]
        for subset in ["audio", "audio-video", "audio-video-bio", "audio-video-ecg"]
        for split in ["a", "b", "c", "d", "e"]
    ]
    DEFAULT_WRITER_BATCH_SIZE = 50

    def _info(self):
        features = {
            "id": datasets.Value("string"),
            "audio": datasets.Audio(sampling_rate=self.target_fs),
            "start": datasets.Value("int32"),
            "end": datasets.Value("int32"),
            "pid": datasets.Value("int32"),
            "target": datasets.ClassLabel(num_classes=79, names=["7","8","9","10","11","13","14","15","16","17","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","40","41","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90"]),
        }
        # Add subset-specific features
        if "video" in self.config.subset:
            features["video"] = datasets.Value("string")
        if "bio" in self.config.subset:
            features["bio_signals"] = datasets.Value("string")
        if "ecg" in self.config.subset:
            features["ecg"] = datasets.Value("string")

        return datasets.DatasetInfo(
            description=f"{self.config.target_type} dataset with {self.config.subset} subset.",
            features=datasets.Features(features),
            homepage="https://speech.fit.vut.cz/software/besst",
            citation="""@article{Pesan2024,
                      title = {Speech production under stress for machine learning: multimodal dataset of 79 cases and 8 signals},
                      volume = {11},
                      ISSN = {2052-4463},
                      url = {http://dx.doi.org/10.1038/s41597-024-03991-w},
                      DOI = {10.1038/s41597-024-03991-w},
                      number = {1},
                      journal = {Scientific Data},
                      publisher = {Springer Science and Business Media LLC},
                      author = {Pešán,  Jan and Juřík,  Vojtěch and Ružičková,  Alexandra and others},
                      year = {2024},
                      month = nov
                    }""",
        )

    def _split_generators(self, dl_manager):
        """
        Caches metadata from the local repository and sets up raw data directory.
        """
        raw_data_dir = self.config.data_dir
        if not raw_data_dir or not os.path.exists(raw_data_dir):
            raise FileNotFoundError(
                f"⚠️ Raw data directory not found: {raw_data_dir}.\n"
                "Please download the dataset manually and provide the correct `data_dir` path.\n"
                "For instructions, visit: https://speech.fit.vut.cz/software/besst"
            )


        # ✅ Define cache location for metadata
        cached_metadata_dir = os.path.join(dl_manager.download_config.cache_dir, "metadata")

        # ✅ Copy metadata to cache if not already cached
        if not os.path.exists(cached_metadata_dir):
            shutil.copytree(self.config.metadata_dir, cached_metadata_dir)

        # Ensure correct metadata path
        partition_dir = os.path.join(
            cached_metadata_dir, "lists", self.config.target_type, self.config.subset, self.config.split_variant
        )

        if not os.path.exists(partition_dir):
            raise FileNotFoundError(f"❌ Metadata directory does not exist: {partition_dir}")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train", "metadata_dir": partition_dir, "raw_data_dir": raw_data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": "validation", "metadata_dir": partition_dir, "raw_data_dir": raw_data_dir},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": "test", "metadata_dir": partition_dir, "raw_data_dir": raw_data_dir},
            ),
        ]

    def _generate_examples(self, split, metadata_dir, raw_data_dir):
        """
        Reads metadata from `.scp` files and links raw data paths.
        """
        current_pid = None
        f = None
        metadata_file = os.path.join(metadata_dir, f"{split}.scp")

        speaker_stats_file = os.path.join(self.config.metadata_dir, "stats/", f"{self.config.target_type}_{self.config.subset}-{self.config.split_variant}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"❌ Missing metadata file: {metadata_file}")

        try:
            split_list = np.loadtxt(metadata_file, dtype=object, delimiter=";", skiprows=1)
        except Exception as e:
            raise RuntimeError(f"❌ Error reading {metadata_file}: {e}")

        if self.speaker_normalization:
            with open(speaker_stats_file, 'r') as fp:
                speaker_stats = json.load(fp);

        for row in split_list:
            pid, segid, start, end, _, segment_name, _, _, _, cognitive_load_label = row
            audio_path = os.path.join(raw_data_dir, "audio", pid, "close_talk.wav")
            # data, sr = sf.read(audio_path, dtype="float32")
            start_ms = int(start)
            end_ms = int(end)

            if pid != current_pid:
                if f is not None:
                    f.close()
                f = sf.SoundFile(audio_path, mode="r")
                current_pid = pid

                # compute frame indices and read only that window
            s = int(start_ms * f.samplerate / 1000)
            n_frames = int((end_ms - start_ms) * f.samplerate / 1000)
            f.seek(s)
            segment = f.read(n_frames)

            if self.speaker_normalization:
                segment = (segment - speaker_stats[pid][0]) / speaker_stats[pid][1]

            # 3) Resample if needed
            if f.samplerate  != self.target_fs:
                segment = librosa.resample(segment,orig_sr=f.samplerate,target_sr=self.target_fs)

            yield f"{pid}_{segid}", {
                "id": f"{pid}_{segid}",
                "audio": {
                    "array": segment,
                    "sampling_rate": self.target_fs,
                },
                "start": start,
                "end": end,
                "pid": int(pid),
                "target": pid,
                "video": os.path.join(raw_data_dir, "video", segment_name) if "video" in self.config.subset else None,
                "bio_signals": os.path.join(raw_data_dir, "bio", segment_name) if "bio" in self.config.subset else None,
                "ecg": os.path.join(raw_data_dir, "ecg", segment_name) if "ecg" in self.config.subset else None,
            }


def parse_args() -> Namespace:
    """
    Parse command-line arguments.
    """
    parser = ArgumentParser(description="Preprocessing script to generate ASR dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to dataset class.")
    parser.add_argument("data_dir", type=str, help="Path to raw data directory (must be downloaded manually).")
    parser.add_argument("dataset_cache", type=str, help="Path to cache directory for storing dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = datasets.load_dataset(
        args.dataset_path,
        keep_in_memory=False,
        cache_dir=args.dataset_cache,
        data_dir=args.data_dir,
        num_proc=4,
    )
