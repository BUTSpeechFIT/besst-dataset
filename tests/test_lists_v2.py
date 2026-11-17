"""
The v1.1 lists (`cognitive-load-block`, `cognitive-load-item`; utils/create_lists_v7.py) are checked against
speech-segments.csv and participants.csv, not against the generator: rows, participants, item-disjointness, the declared
test and validation items, the label rules, class coverage, the symlinked shared test list, the builders' configs, and that
the committed files are exactly what the generator writes.  python tests/test_lists_v2.py  (no audio needed)
"""
import csv
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "dataset" / "metadata"
FOLDS = "abcde"
ITEM_RE = re.compile(r"^R([1-4])_(\d)$")
TEST_ITEMS = {"R2_0", "R2_3", "R3_1", "R3_4", "R4_0", "R4_1", "R4_5"}
VALIDATION = {
    "cognitive-load-block": {"a": {"R1_0", "R3_2", "R4_4"}, "b": {"R2_2", "R3_0", "R4_3"}, "c": {"R1_1", "R3_3", "R4_2"},
                             "d": {"R2_1", "R3_2", "R4_6"}, "e": {"R1_2", "R3_0", "R4_4"}},
    "cognitive-load-item": {"a": {"R1_1", "R3_3", "R4_2"}, "b": {"R2_1", "R3_2", "R4_3"}, "c": {"R1_0", "R2_2", "R4_4"},
                            "d": {"R1_2", "R3_2", "R4_6"}, "e": {"R3_0", "R2_2", "R4_4"}},
}


def label_block(item):
    return {"1": "1", "2": "1", "3": "2", "4": "3"}[ITEM_RE.match(item).group(1)]


def label_item(item):
    k = int(ITEM_RE.match(item).group(2))
    return "1" if k == 0 else ("2" if k <= 2 else "3")


LABEL = {"cognitive-load-block": label_block, "cognitive-load-item": label_item}


def universe():
    """(pid, sid) -> csv row for the Rebus items of R1-R4 of the participants that have every modality."""
    with open(META / "participants.csv", encoding="utf-8", newline="") as f:
        rd = csv.reader(f, delimiter=";"); next(rd)
        pids = {r[0] for r in rd if all(x == "1" for x in r[6:])}
    rows = {}
    with open(META / "speech-segments.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["pid"] in pids and r["phase_name"] == "Rebus" and ITEM_RE.match(r["phase"]):
                rows[(r["pid"], r["sid"])] = r
    return pids, rows


def read_scp(path):
    with open(path, encoding="utf-8", newline="") as f:
        return [line.rstrip("\n").split(";") for line in f if line.strip()]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class ListsV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pids, cls.rows = universe()

    def test_universe_is_the_v1_universe(self):
        self.assertEqual(len(self.pids), 71)
        self.assertEqual(len(self.rows), 1448)

    def check_rows_against_csv(self, variant, rows):
        label_of = LABEL[variant]
        for r in rows:
            self.assertEqual(len(r), 10, r)
            src = self.rows[(r[0], r[1])]
            self.assertEqual(r[2:9], [src["start"], src["end"], src["text"], src["phase"], src["phase_name"],
                                      src["subjective_load"], src["rebus_score"]])
            self.assertEqual(r[9], label_of(r[5]), r)

    def test_each_variant(self):
        for variant in VALIDATION:
            with self.subTest(variant=variant):
                base = META / "lists" / variant / "audio"
                test = read_scp(base / "test.scp")
                self.check_rows_against_csv(variant, test)
                self.assertEqual(len(test), 546)
                self.assertEqual({r[5] for r in test}, TEST_ITEMS)
                # csv order kept
                order = [k for k, v in self.rows.items() if v["phase"] in TEST_ITEMS]
                self.assertEqual([(r[0], r[1]) for r in test], order)
                items_per_class = defaultdict(set)
                for r in test:
                    items_per_class[r[9]].add(r[5])
                self.assertEqual(sorted(items_per_class), ["1", "2", "3"])
                self.assertTrue(all(len(v) >= 2 for v in items_per_class.values()), dict(items_per_class))
                for fold in FOLDS:
                    link = base / fold / "test.scp"
                    self.assertTrue(link.is_symlink() and os.readlink(link) == "../test.scp", link)
                    self.assertEqual(read_scp(link), test)
                    train, val = read_scp(base / fold / "train.scp"), read_scp(base / fold / "validation.scp")
                    self.check_rows_against_csv(variant, train); self.check_rows_against_csv(variant, val)
                    keys = [{(r[0], r[1]) for r in s} for s in (train, val, test)]
                    self.assertEqual(sum(len(k) for k in keys), len(keys[0] | keys[1] | keys[2]), "splits overlap")
                    self.assertEqual(keys[0] | keys[1] | keys[2], set(self.rows), "splits do not partition the corpus")
                    items = [{r[5] for r in s} for s in (train, val, test)]
                    self.assertFalse(items[0] & items[1] or items[0] & items[2] or items[1] & items[2], "items shared")
                    self.assertEqual(items[1], VALIDATION[variant][fold])
                    for s in (train, val, test):
                        self.assertEqual({r[0] for r in s}, self.pids, "every participant in every split")
                    train_items = defaultdict(set)
                    for r in train:
                        train_items[r[9]].add(r[5])
                    self.assertEqual(sorted(train_items), ["1", "2", "3"])
                    if variant == "cognitive-load-block":
                        self.assertTrue(all(len(v) >= 2 for v in train_items.values()), dict(train_items))
                    else:  # only R1_0 and R3_0 carry k=0 outside the test set
                        self.assertEqual(len(train_items["1"]), 1 if fold in "ce" else 2, (fold, dict(train_items)))

    def test_variants_share_segments_and_pool(self):
        b, i = (META / "lists" / v / "audio" for v in ("cognitive-load-block", "cognitive-load-item"))
        tb, ti = read_scp(b / "test.scp"), read_scp(i / "test.scp")
        self.assertEqual([r[:9] for r in tb], [r[:9] for r in ti])
        self.assertNotEqual([r[9] for r in tb], [r[9] for r in ti])
        for fold in FOLDS:
            pool = lambda base: {(r[0], r[1]) for s in ("train", "validation") for r in read_scp(base / fold / f"{s}.scp")}
            self.assertEqual(pool(b), pool(i))

    def test_builders_know_the_variants_and_stats_exist(self):
        for script in ("dataset.py", "speaker_dataset.py", "gender_dataset.py"):
            mod = load_module(ROOT / "dataset" / script, "besst_" + script[:-3])
            cls = next(v for v in vars(mod).values() if isinstance(v, type) and hasattr(v, "BUILDER_CONFIGS"))
            names = {c.name: c for c in cls.BUILDER_CONFIGS}
            for variant in VALIDATION:
                for fold in FOLDS:
                    name = f"{variant}_audio-{fold}"
                    self.assertIn(name, names, script)
                    self.assertEqual(names[name].target_type, variant)
                    self.assertEqual(names[name].split_variant, fold)
        for variant in VALIDATION:
            for fold in FOLDS:
                self.assertTrue((META / "stats" / f"{variant}_audio-{fold}").is_file())

    def test_committed_lists_equal_a_fresh_generation(self):
        gen = load_module(ROOT / "utils" / "create_lists_v7.py", "create_lists_v7")
        tmp = Path(tempfile.mkdtemp(prefix="besst-lists-v2."))
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        gen.create(str(META), str(tmp), copy_stats=False, verbose=False)
        for variant in VALIDATION:
            for rel in ["test.scp"] + [f"{f}/{s}.scp" for f in FOLDS for s in ("train", "validation")]:
                a, b = META / "lists" / variant / "audio" / rel, tmp / "lists" / variant / "audio" / rel
                self.assertEqual(a.read_bytes(), b.read_bytes(), rel)
            for f in FOLDS:
                self.assertEqual(os.readlink(tmp / "lists" / variant / "audio" / f / "test.scp"), "../test.scp")


if __name__ == "__main__":
    unittest.main()
