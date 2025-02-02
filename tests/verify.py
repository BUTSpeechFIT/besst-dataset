import os

import numpy as np
from datasets import load_dataset

# Define paths
ROOT_DIR = os.path.dirname(__file__)
METADATA_DIR = os.path.join(ROOT_DIR, "metadata/lists")
EXPECTED_VERSIONS = ["cognitive-load", "physical-load"]
EXPECTED_SUBSETS = ["audio", "audio-video", "audio-video-bio", "audio-video-ecg"]
SPLIT_VARIANTS = ["a", "b", "c", "d", "e"]
SPLITS = ["train", "test", "validation"]


def check_metadata():
    """Verify metadata .scp files exist and are readable."""
    print("\n🔍 Checking metadata integrity...")

    missing_files = []

    for version in EXPECTED_VERSIONS:
        for subset in EXPECTED_SUBSETS:
            for split_variant in SPLIT_VARIANTS:
                for split in SPLITS:
                    metadata_path = os.path.join(
                        METADATA_DIR, version, subset, split_variant, f"{split}.scp"
                    )
                    if not os.path.exists(metadata_path):
                        missing_files.append(metadata_path)

    if missing_files:
        print(f"❌ Missing metadata files:\n" + "\n".join(missing_files))
    else:
        print("✅ All metadata files are present.")

    # Check if .scp files are empty
    for version in EXPECTED_VERSIONS:
        for subset in EXPECTED_SUBSETS:
            for split_variant in SPLIT_VARIANTS:
                for split in SPLITS:
                    metadata_path = os.path.join(
                        METADATA_DIR, version, subset, split_variant, f"{split}.scp"
                    )
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            lines = f.readlines()
                            if not lines:
                                print(f"⚠️ {metadata_path} is empty.")


def check_raw_data(data_dir):
    """Verify the existence of raw data files."""
    print("\n🔍 Checking raw data integrity...")

    if not os.path.exists(data_dir):
        print(f"❌ Raw data directory does not exist: {data_dir}")
        return

    missing_files = []

    for version in EXPECTED_VERSIONS:
        for subset in EXPECTED_SUBSETS:
            for split_variant in SPLIT_VARIANTS:
                for split in SPLITS:
                    metadata_path = os.path.join(
                        METADATA_DIR, version, subset, split_variant, f"{split}.scp"
                    )

                    if os.path.exists(metadata_path):
                        try:
                            split_list = np.loadtxt(metadata_path, dtype=object, delimiter=";", skiprows=1)
                        except Exception as e:
                            print(f"❌ Error reading {metadata_path}: {e}")
                            continue

                        # Check if referenced files exist
                        for row in split_list:
                            pid, segid, start, end, _, segment_name, _, _, _, _ = row
                            audio_file = os.path.join(data_dir, "audio", pid, "close_talk.wav")

                            if not os.path.exists(audio_file):
                                missing_files.append(audio_file)

    if missing_files:
        print(f"❌ Missing raw data files:\n" + "\n".join(missing_files))
    else:
        print("✅ All raw data files are present.")


def test_dataset_loading(data_dir):
    """Try loading the dataset to verify it works."""
    print("\n🔍 Testing dataset loading...")

    try:
        dataset = load_dataset(
            "dataset/besst.py",
            name="cognitive-load_audio_a",
            data_dir=data_dir,
            cache_dir="./cache",
            metadata_dir=os.path.abspath("dataset/metadata"),
            trust_remote_code=True,  # Ensures it reads from the local script
        )
        print("✅ Dataset loaded successfully!")

        # Print sample
        sample = dataset["train"][0]
        print("\n🔍 Sample data:")
        for key, value in sample.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")


if __name__ == "__main__":
    # data_dir = input("\n📂 Enter the path to the raw data directory: ").strip()
    data_dir = "/mnt/matylda3/ipesan/EXP/besst-process/dataset/v5/raw/"
    check_metadata()
    check_raw_data(data_dir)
    test_dataset_loading(data_dir)

    print("\n🎯 Dataset verification completed.")
