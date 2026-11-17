#!/usr/bin/env python3
"""
Create the BESST v2 cognitive-load lists: ONE item-disjoint split, TWO label variants (besst-dataset v1.1).

Why. The v1.0 lists (`lists/cognitive-load`, utils/create_lists_v6.py) label a segment by the item index k within its
Rebus block (k 0-1 low, 2-3 medium, 4-6 high). Only four items carry "high" (R3_4, R4_4, R4_5, R4_6); two of them are
the shared test set, so every fold trains "high" on ONE sentence and tests it on two others (spans 5 and 6, never seen
in training). Besides, "high" exists only in the last two blocks, and the test sentences of the three classes differ in
length (8-9 s low, 12-14 s high). A HuBERT-MHFA sweep on those lists fit the training set and stayed at chance on the test
set (besst repo, RESEARCH_LOG Session 14).

What. Both variants use the same participants (the 71 with every modality, as v1.0), the same segments (every Rebus
segment of blocks R1-R4; R0 is the learning phase and stays out), the same shared test items and the same train+validation
pool per fold; they differ in the label and, because the label decides what a validation item is, in the validation items.

  variant                 label                                           items per class (corpus / shared test)
  cognitive-load-block    1 = R1+R2, 2 = R3, 3 = R4 (block = span of the   7 / 5 / 7   (test 2 / 2 / 3)
                          reading-span block: 3+4, 5, 7 passages)
  cognitive-load-item     1 = k 0, 2 = k 1-2, 3 = k 3-6 (k = passages      4 / 8 / 7   (test 2 / 2 / 3)
                          already read in the block = final words held)

Shared test items: R2_0 R2_3 R3_1 R3_4 R4_0 R4_1 R4_5 (546 segments). They were chosen so that under BOTH labels every
class has at least two test sentences and the class mean durations are close (block 9.4 / 9.7 / 10.2 s, item 9.8 / 9.3 /
10.3 s, against 8.7 / 10.6 / 13.0 s in v1.0). Validation: three items per fold, one per class where the pool allows it;
the item variant has only two non-test k=0 items (R1_0, R3_0), so folds a, b and d validate on classes 2 and 3 only and
keep both k=0 items in training, while folds c and e validate on one k=0 item and train on the other (the one-item
condition of v1.0, kept on purpose so that it can be compared).

Speaker stats (`stats/<variant>_audio-<fold>`, mean/std per speaker for speaker_normalization=True) are copied from the
v1.0 fold of the same letter: they are raw-audio statistics per speaker over the fold's training segments and vary by a
few percent between folds (median 2.6 % of the std across the five v1.0 folds). Re-estimate them with
utils/estimate_speaker_stats.py when the raw audio is at hand.

Usage (from the repository root):
  python utils/create_lists_v7.py                       # writes under dataset/metadata
  python utils/create_lists_v7.py --output_dir /tmp/x   # elsewhere (stats are copied from --metadata_dir)
"""
import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

FOLDS = ["a", "b", "c", "d", "e"]
BLOCK_ITEMS = {1: 3, 2: 4, 3: 5, 4: 7}  # passages per block R1..R4 (R0 = 2, excluded)
ITEMS = [f"R{b}_{k}" for b, n in BLOCK_ITEMS.items() for k in range(n)]  # 19 items, R1_0 .. R4_6
TEST_ITEMS = ["R2_0", "R2_3", "R3_1", "R3_4", "R4_0", "R4_1", "R4_5"]
SOURCE_STATS = "cognitive-load"  # v1.0 variant whose per-fold speaker stats are copied


def block(item):
    return int(item[1])


def k_index(item):
    return int(item.split("_")[1])


def label_block(item):
    return {1: "1", 2: "1", 3: "2", 4: "3"}[block(item)]


def label_item(item):
    k = k_index(item)
    return "1" if k == 0 else ("2" if k <= 2 else "3")


VARIANTS = {
    "cognitive-load-block": {
        "label": label_block,
        "validation": {"a": ["R1_0", "R3_2", "R4_4"], "b": ["R2_2", "R3_0", "R4_3"], "c": ["R1_1", "R3_3", "R4_2"],
                       "d": ["R2_1", "R3_2", "R4_6"], "e": ["R1_2", "R3_0", "R4_4"]},
    },
    "cognitive-load-item": {
        "label": label_item,
        "validation": {"a": ["R1_1", "R3_3", "R4_2"], "b": ["R2_1", "R3_2", "R4_3"], "c": ["R1_0", "R2_2", "R4_4"],
                       "d": ["R1_2", "R3_2", "R4_6"], "e": ["R3_0", "R2_2", "R4_4"]},
    },
}


def load_participants(metadata_dir):
    """The participants with every modality flag set (columns 7.. of participants.csv), as utils/create_lists_v6.py."""
    table = np.loadtxt(Path(metadata_dir) / "participants.csv", dtype=object, delimiter=";", skiprows=1)
    return set(table[np.all(table[:, 6:].astype(int), axis=1), 0])


def load_segments(metadata_dir, pids):
    """Rows of speech-segments.csv (csv order) of the given participants that are Rebus items of blocks R1-R4."""
    rows = []
    with open(Path(metadata_dir) / "speech-segments.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row["pid"] in pids and row["phase_name"] == "Rebus" and row["phase"] in ITEMS:
                rows.append(row)
    return rows


def scp_line(row, label):
    """`pid;segid;start;end;text;segment_name;phase;subjective_load;rebus_score;label` (the v1.0 column order)."""
    return ";".join([row["pid"], row["sid"], row["start"], row["end"], row["text"], row["phase"], row["phase_name"],
                     row["subjective_load"], row["rebus_score"], label])


def write_scp(path, rows, label_of):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(scp_line(row, label_of(row["phase"])) + "\n")


def summarize(rows, label_of):
    by = {}
    for row in rows:
        by.setdefault(label_of(row["phase"]), Counter())[row["phase"]] += 1
    return " | ".join("L{}: {}".format(lab, " ".join("{}({})".format(i, n) for i, n in sorted(by[lab].items())))
                      for lab in sorted(by))


def create(metadata_dir, output_dir, copy_stats=True, verbose=True):
    pids = load_participants(metadata_dir)
    segments = load_segments(metadata_dir, pids)
    assert {r["pid"] for r in segments} == pids, "every listed participant must have Rebus segments"
    test_rows = [r for r in segments if r["phase"] in TEST_ITEMS]
    for variant, spec in VARIANTS.items():
        label_of = spec["label"]
        base = Path(output_dir) / "lists" / variant / "audio"
        write_scp(base / "test.scp", test_rows, label_of)
        if verbose:
            print("\n{}  ({} participants, {} segments, test {})".format(variant, len(pids), len(segments), len(test_rows)))
            print("  test        n={:4d} | {}".format(len(test_rows), summarize(test_rows, label_of)))
        for fold in FOLDS:
            val_items = set(spec["validation"][fold])
            assert not val_items & set(TEST_ITEMS) and val_items <= set(ITEMS)
            val_rows = [r for r in segments if r["phase"] in val_items]
            train_rows = [r for r in segments if r["phase"] not in val_items and r["phase"] not in TEST_ITEMS]
            write_scp(base / fold / "train.scp", train_rows, label_of)
            write_scp(base / fold / "validation.scp", val_rows, label_of)
            link = base / fold / "test.scp"
            if link.is_symlink() or link.exists():
                link.unlink()
            os.symlink("../test.scp", link)
            if verbose:
                print("  {} train     n={:4d} | {}".format(fold, len(train_rows), summarize(train_rows, label_of)))
                print("  {} validation n={:4d} | {}".format(fold, len(val_rows), summarize(val_rows, label_of)))
            if copy_stats:
                src = Path(metadata_dir) / "stats" / "{}_audio-{}".format(SOURCE_STATS, fold)
                dst = Path(output_dir) / "stats" / "{}_audio-{}".format(variant, fold)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
    return len(segments), len(test_rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata_dir", default="dataset/metadata", help="holds speech-segments.csv, participants.csv, stats/")
    ap.add_argument("--output_dir", default="dataset/metadata", help="lists/<variant>/audio/... and stats/ are written here")
    ap.add_argument("--no_stats", action="store_true", help="do not copy the v1.0 speaker stats")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    n, t = create(a.metadata_dir, a.output_dir, copy_stats=not a.no_stats, verbose=not a.quiet)
    if not a.quiet:
        print("\nwrote {} under {}: {} segments, {} in the shared test set".format(", ".join(VARIANTS), a.output_dir, n, t))


if __name__ == "__main__":
    main()
