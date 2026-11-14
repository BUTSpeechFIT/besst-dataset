"""The HF cache key of a BESST config must change whenever the data it describes changes.

Regression test for the stale-list bug (v0.5 lists served from the HF cache under a v0.6.0 install):
the config id ignored the lists, so an upgraded package kept reading an old cached copy of them.

Run with the project environment (datasets >= 2.20):  python -m unittest discover -s tests
No audio is needed: only the builder is constructed, never run.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import datasets

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = str(ROOT / "dataset" / "dataset.py")
GENDER_SCRIPT = str(ROOT / "dataset" / "gender_dataset.py")
META = ROOT / "dataset" / "metadata"
CONFIG = "cognitive-load_audio-a"


def copy_metadata(dest):
    """Only what the config under test needs (lists of fold a, its stats, participants)."""
    audio = Path("lists") / "cognitive-load" / "audio"
    for rel in [audio / "a" / "train.scp", audio / "a" / "validation.scp", audio / "test.scp",
                Path("stats") / "cognitive-load_audio-a", Path("participants.csv")]:
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(META / rel, dest / rel)
    # same layout as the package: the shared test list is linked into every fold directory
    os.symlink("../test.scp", dest / audio / "a" / "test.scp")
    return dest


class ConfigIdTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="besst-config-id."))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.data_dir = self.tmp / "raw"
        self.data_dir.mkdir()
        self.cache = self.tmp / "hf-cache"

    def builder(self, metadata_dir, script=SCRIPT, **kwargs):
        kw = dict(target_fs=16000, speaker_normalization=True)
        kw.update(kwargs)
        return datasets.load_dataset_builder(script, name=CONFIG, data_dir=str(self.data_dir),
                                             metadata_dir=str(metadata_dir), cache_dir=str(self.cache),
                                             trust_remote_code=True, **kw)

    def test_same_lists_in_two_locations_share_one_id(self):
        a = copy_metadata(self.tmp / "meta_a")
        b = copy_metadata(self.tmp / "meta_b")
        self.assertEqual(self.builder(a).config_id, self.builder(b).config_id)

    def test_changed_list_changes_the_id(self):
        a = copy_metadata(self.tmp / "meta_a")
        b = copy_metadata(self.tmp / "meta_b")
        train = b / "lists" / "cognitive-load" / "audio" / "a" / "train.scp"
        lines = train.read_text(encoding="utf-8").splitlines(True)
        train.write_text("".join(lines[1:]), encoding="utf-8")          # drop one training segment
        self.assertNotEqual(self.builder(a).config_id, self.builder(b).config_id)

    def test_changed_test_list_or_stats_changes_the_id(self):
        a = copy_metadata(self.tmp / "meta_a")
        base = self.builder(a).config_id
        b = copy_metadata(self.tmp / "meta_b")
        test = b / "lists" / "cognitive-load" / "audio" / "a" / "test.scp"
        test.write_text(test.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assertNotEqual(base, self.builder(b).config_id)
        c = copy_metadata(self.tmp / "meta_c")
        stats = c / "stats" / "cognitive-load_audio-a"
        stats.write_text(stats.read_text(encoding="utf-8").replace("]", " ]", 1), encoding="utf-8")
        self.assertNotEqual(base, self.builder(c).config_id)

    def test_audio_options_are_part_of_the_id(self):
        a = copy_metadata(self.tmp / "meta_a")
        base = self.builder(a).config_id
        self.assertNotEqual(base, self.builder(a, target_fs=44100).config_id)
        self.assertNotEqual(base, self.builder(a, speaker_normalization=False).config_id)
        self.assertEqual(base, self.builder(a).config_id)

    def test_builder_reads_options_from_its_config(self):
        a = copy_metadata(self.tmp / "meta_a")
        b = self.builder(a, target_fs=8000, speaker_normalization=False)
        self.assertEqual(b.target_fs, 8000)
        self.assertFalse(b.speaker_normalization)
        self.assertEqual(b.info.features["audio"].sampling_rate, 8000)
        self.assertEqual(b.config.lists_dir(), str(a / "lists" / "cognitive-load" / "audio" / "a"))
        self.assertTrue(os.path.isfile(b.config.stats_file()))

    def test_gender_labels_file_is_part_of_the_id(self):
        a = copy_metadata(self.tmp / "meta_a")
        b = copy_metadata(self.tmp / "meta_b")
        participants = b / "participants.csv"
        participants.write_text(participants.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assertNotEqual(self.builder(a, script=GENDER_SCRIPT).config_id,
                            self.builder(b, script=GENDER_SCRIPT).config_id)
        # the cognitive-load builder does not depend on participants.csv
        self.assertEqual(self.builder(a).config_id, self.builder(b).config_id)


if __name__ == "__main__":
    unittest.main()
