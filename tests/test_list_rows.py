"""
Every builder must yield exactly one example per list row: the `.scp` lists have no header, so no builder may skip the first
row (v1.0.2's speaker builder did, dropping `7_16` / `7_15` / `7_22` from the fold-a splits). Runs the generators on a tiny
synthetic corpus (two participants, 48 kHz noise), no Hugging Face cache involved.
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import io

import datasets
import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "cognitive_load": str(ROOT / "dataset" / "dataset.py"),
    "speaker": str(ROOT / "dataset" / "speaker_dataset.py"),
    "gender": str(ROOT / "dataset" / "gender_dataset.py"),
}
CONFIG = "cognitive-load_audio-a"
SR = 48000

# pid;segid;start;end;text;segment_name;phase;subjective_load;rebus_score;label  (as in the released lists)
ROWS = {
    "train": [("7", "16", 100, 1300, "R1_1", "1"), ("7", "17", 1400, 2600, "R1_2", "2"), ("8", "5", 200, 1500, "R3_4", "3")],
    "validation": [("7", "21", 300, 1200, "R2_1", "1"), ("8", "9", 100, 900, "R2_3", "2")],
    "test": [("7", "22", 500, 1700, "R4_5", "3"), ("8", "12", 1800, 2900, "R4_6", "3")],
}


def write_corpus(root):
    meta, raw = root / "metadata", root / "raw"
    fold = meta / "lists" / "cognitive-load" / "audio" / "a"
    fold.mkdir(parents=True)
    for split, rows in ROWS.items():
        path = fold / f"{split}.scp" if split != "test" else fold.parent / "test.scp"
        with open(path, "w") as fp:
            for pid, segid, start, end, phase, label in rows:
                fp.write(";".join([pid, segid, str(start), str(end), "SOME TEXT", f"{pid}_{segid}_x.wav", phase, "4.0", "0.5", label]) + "\n")
    os.symlink("../test.scp", fold / "test.scp")
    (meta / "stats").mkdir()
    with open(meta / "stats" / CONFIG, "w") as fp:
        json.dump({"7": [0.0, 0.01, SR * 3], "8": [0.0, 0.01, SR * 3]}, fp)
    with open(meta / "participants.csv", "w") as fp:
        fp.write("pid;gender;PSS14\n7;M;29\n8;F;26\n")
    rng = np.random.default_rng(0)
    for pid in ("7", "8"):
        (raw / "audio" / pid).mkdir(parents=True)
        sf.write(raw / "audio" / pid / "close_talk.wav", (0.01 * rng.standard_normal(SR * 3)).astype("float32"), SR)
    return meta, raw


class ListRowsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="besst-list-rows."))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.meta, self.raw = write_corpus(self.tmp)

    def examples(self, target, split):
        builder = datasets.load_dataset_builder(SCRIPTS[target], name=CONFIG, data_dir=str(self.raw),
                                                metadata_dir=str(self.meta), cache_dir=str(self.tmp / "hf-cache"),
                                                trust_remote_code=True, target_fs=16000, speaker_normalization=True)
        return list(builder._generate_examples(split, builder.config.lists_dir(), str(self.raw)))

    def test_every_builder_yields_one_example_per_list_row(self):
        for target in SCRIPTS:
            for split, rows in ROWS.items():
                with self.subTest(target=target, split=split):
                    got = self.examples(target, split)
                    self.assertEqual([key for key, _ in got], [f"{pid}_{segid}" for pid, segid, *_ in rows])

    def test_targets_of_the_first_row(self):
        first = {target: self.examples(target, "train")[0][1] for target in SCRIPTS}
        self.assertEqual(first["cognitive_load"]["id"], "7_16")
        self.assertEqual(first["cognitive_load"]["target"], "low")  # label 1 -> low (ClassLabel encodes later)
        self.assertEqual(first["speaker"]["target"], "7")
        self.assertEqual(first["gender"]["target"], "M")
        n = int((1300 - 100) / 1000 * 16000)
        audio, sr = sf.read(io.BytesIO(first["speaker"]["audio"]["bytes"]))
        self.assertEqual(len(audio), n)
        self.assertEqual(sr, 16000)

    def test_speaker_normalised_audio_is_not_clipped(self):
        """v1.0.3 and earlier stored arrays as 16-bit PCM, so unit-variance audio was cut at +-1 (v1.0.4 fix).
        Through the full datasets path (encode, Arrow cache, decode) the stored audio must equal the builder's floats."""
        stats = json.loads((self.meta / "stats" / CONFIG).read_text())
        for target in SCRIPTS:
            with self.subTest(target=target):
                ds = datasets.load_dataset(SCRIPTS[target], name=CONFIG, data_dir=str(self.raw), metadata_dir=str(self.meta),
                                           cache_dir=str(self.tmp / f"hf-{target}"), trust_remote_code=True, target_fs=16000,
                                           speaker_normalization=True)
                ex = ds["train"][0]
                arr = np.asarray(ex["audio"]["array"])
                self.assertGreater(np.abs(arr).max(), 1.5, "speaker-normalised audio still clipped at 1")
                # equal to the z-scored, resampled source window
                raw, sr = sf.read(self.raw / "audio" / "7" / "close_talk.wav", dtype="float64")
                seg = raw[int(100 * sr / 1000): int(100 * sr / 1000) + int(1200 * sr / 1000)]
                ref = librosa.resample((seg - stats["7"][0]) / stats["7"][1], orig_sr=sr, target_sr=16000)
                np.testing.assert_allclose(arr, ref, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
