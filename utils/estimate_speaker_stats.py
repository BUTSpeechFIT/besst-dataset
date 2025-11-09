import os

import numpy as np
from collections import defaultdict
from datasets import load_dataset, Audio

ROOT_DIR = os.path.dirname(__file__)
#METADATA_DIR = os.path.join(ROOT_DIR, "metadata/lists")
EXPECTED_VERSIONS = ["cognitive-load", "physical-load"]
SPLIT_VARIANTS = ["a", "b", "c", "d", "e"]
SPLITS = ["train", "test", "validation"]


def generate_speaker_stats(dataset_package_path, data_dir):
    """Verify metadata .scp files exist and are readable.
    :param path:
    """
    print("\n🔍 Checking metadata integrity...")



    for version in EXPECTED_VERSIONS:
        for split_variant in SPLIT_VARIANTS:
            dataset_name = f"{version}_audio-{split_variant}"
            dataset = load_dataset(
                f"{dataset_package_path}/dataset.py",
                name=dataset_name,
                data_dir=data_dir,
                metadata_dir=f"{dataset_package_path}/metadata",
                trust_remote_code=True,  # Ensures it reads from the local script
            )
            # Make sure audio is decoded:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

            # 2) accumulate per-speaker samples
            buff = defaultdict(list)
            for ex in dataset["train"]:
                spk = ex["pid"]
                arr = ex["audio"]["array"]
                buff[spk].append(arr)

            # 3) compute stats
            speaker_stats = {
                spk: (np.concatenate(arrs).mean(), np.concatenate(arrs).std(), len(np.concatenate(arrs)))
                for spk, arrs in buff.items()
            }

            print(speaker_stats)


if __name__ == "__main__":
    # data_dir = input("\n📂 Enter the path to the raw data directory: ").strip()
    data_dir = "/mnt/matylda3/ipesan/EXP/besst-process/dataset/v5/raw/"
    dataset_package_path="/homes/kazi/ipesan/devel/python/besst-dataset/dataset/"
    generate_speaker_stats(dataset_package_path, data_dir)


#
# # save to disk so you can reload in training:
# import pickle
# with open("speaker_stats.pkl", "wb") as f:
#     pickle.dump(speaker_stats, f)