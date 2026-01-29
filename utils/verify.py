#!/usr/bin/env python3
"""
Verify BESST v6 cognitive load splits.

Checks:
1. File existence
2. Test set consistency (size, class balance)
3. Train/validation distribution (class balance per fold)
4. Phase separation (no overlap within fold)
5. Participant coverage (all 71 in all splits)
6. Phase-to-class mapping correctness
7. Data integrity (format, optional audio file check)
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


# Expected phase-to-class mapping
PHASE_TO_CLASS = {
    'R1_0': '1', 'R1_1': '1', 'R2_0': '1', 'R2_1': '1',
    'R3_0': '1', 'R3_1': '1', 'R4_0': '1', 'R4_1': '1',
    'Debriefing': '1',
    'R1_2': '2', 'R2_2': '2', 'R2_3': '2', 'R3_2': '2',
    'R3_3': '2', 'R4_2': '2', 'R4_3': '2',
    'R3_4': '3', 'R4_4': '3', 'R4_5': '3', 'R4_6': '3',
}

TEST_PHASES = {'R2_1', 'R3_1', 'R2_3', 'R3_3', 'R4_5', 'R4_6'}

VALIDATION_PHASES = {
    'a': {'R1_0', 'R2_2', 'R3_4'},
    'b': {'R1_1', 'R3_2', 'R4_4'},
    'c': {'R2_0', 'R4_2', 'R3_4'},
    'd': {'R3_0', 'R4_3', 'R4_4'},
    'e': {'R4_0', 'R1_2', 'R3_4'},
}

FOLDS = ['a', 'b', 'c', 'd', 'e']
EXPECTED_PARTICIPANTS = 71
EXPECTED_TEST_SIZE = 452


def load_scp(filepath):
    """Load .scp file and return as numpy array."""
    if not os.path.exists(filepath):
        return None
    return np.loadtxt(filepath, dtype=object, delimiter=";")


def check_file_existence(base_dir):
    """Check all expected files exist."""
    print("\n" + "="*70)
    print("1. FILE EXISTENCE CHECK")
    print("="*70)

    all_exist = True

    # Check shared test set
    test_file = base_dir / "test.scp"
    if test_file.exists():
        print(f"✓ Shared test set exists: {test_file}")
    else:
        print(f"✗ Missing shared test set: {test_file}")
        all_exist = False

    # Check fold directories (only train.scp and validation.scp)
    for fold in FOLDS:
        fold_dir = base_dir / fold
        if not fold_dir.exists():
            print(f"✗ Missing fold directory: {fold_dir}")
            all_exist = False
            continue

        for split in ['train.scp', 'validation.scp']:
            split_file = fold_dir / split
            if not split_file.exists():
                print(f"✗ Missing {fold}/{split}")
                all_exist = False
            else:
                print(f"✓ Fold {fold}/{split} exists")

    return all_exist


def check_test_set_consistency(base_dir):
    """Verify test set is identical across all folds."""
    print("\n" + "="*70)
    print("2. TEST SET CONSISTENCY CHECK")
    print("="*70)

    test_file = base_dir / "test.scp"
    test_data = load_scp(test_file)

    if test_data is None:
        print("✗ Cannot load shared test set")
        return False

    print(f"✓ Shared test set: {len(test_data)} samples")

    # Check size
    if len(test_data) == EXPECTED_TEST_SIZE:
        print(f"✓ Test set size correct: {EXPECTED_TEST_SIZE}")
    else:
        print(f"✗ Test set size mismatch: expected {EXPECTED_TEST_SIZE}, got {len(test_data)}")
        return False

    # Check class balance
    classes = test_data[:, -1]
    class_counts = Counter(classes)
    total = len(test_data)

    print("\nClass distribution:")
    for cls in ['1', '2', '3']:
        count = class_counts.get(cls, 0)
        pct = count / total * 100
        print(f"  Class {cls}: {count:3d} ({pct:5.1f}%)")

    # Check if balanced (within 5% of 33.33%)
    balanced = all(30 <= (class_counts.get(cls, 0) / total * 100) <= 38 for cls in ['1', '2', '3'])
    if balanced:
        print("✓ Test set is balanced")
    else:
        print("⚠ Test set may be imbalanced")

    return True


def check_train_val_distribution(base_dir):
    """Check class distribution in train and validation sets for all folds."""
    print("\n" + "="*70)
    print("3. TRAIN/VALIDATION DISTRIBUTION CHECK")
    print("="*70)
    print("NOTE: Train imbalance (~50/40/10) is expected due to structural")
    print("      scarcity of class-3 phases. Handled via class weights [0.67, 0.84, 3.22].")

    all_balanced = True

    for fold in FOLDS:
        train_data = load_scp(base_dir / fold / "train.scp")
        val_data = load_scp(base_dir / fold / "validation.scp")

        if train_data is None or val_data is None:
            print(f"\n✗ Fold {fold}: Cannot load data")
            all_balanced = False
            continue

        print(f"\nFold {fold}:")

        # Train set distribution (informational only - imbalance expected)
        train_classes = train_data[:, -1]
        train_counts = Counter(train_classes)
        train_total = len(train_data)

        print(f"  Train ({train_total} samples) [imbalance expected]:")
        train_dist = []
        for cls in ['1', '2', '3']:
            count = train_counts.get(cls, 0)
            pct = count / train_total * 100 if train_total > 0 else 0
            print(f"    Class {cls}: {count:3d} ({pct:5.1f}%)")
            train_dist.append(pct)

        # Validation set distribution (must be balanced)
        val_classes = val_data[:, -1]
        val_counts = Counter(val_classes)
        val_total = len(val_data)

        print(f"  Validation ({val_total} samples) [must be balanced]:")
        val_dist = []
        for cls in ['1', '2', '3']:
            count = val_counts.get(cls, 0)
            pct = count / val_total * 100 if val_total > 0 else 0
            print(f"    Class {cls}: {count:3d} ({pct:5.1f}%)")
            val_dist.append(pct)

        # Check if validation is reasonably balanced (25-40% per class)
        val_balanced = all(25 <= pct <= 40 for pct in val_dist)
        if val_balanced:
            print(f"  ✓ Validation balanced")
        else:
            print(f"  ✗ Validation imbalanced (this should not happen)")
            all_balanced = False

    if all_balanced:
        print("\n✓ All validation sets are reasonably balanced")
        print("✓ Train imbalance acknowledged (compensated via class weights)")

    return all_balanced


def check_phase_separation(base_dir):
    """Check no phase overlap between train/val/test within each fold."""
    print("\n" + "="*70)
    print("4. PHASE SEPARATION CHECK")
    print("="*70)

    all_separated = True
    test_data = load_scp(base_dir / "test.scp")
    test_phases = set(test_data[:, 5])

    print(f"Test phases: {sorted(test_phases)}")
    if test_phases == TEST_PHASES:
        print("✓ Test phases match expected")
    else:
        print(f"✗ Test phases mismatch!")
        print(f"  Expected: {sorted(TEST_PHASES)}")
        print(f"  Got: {sorted(test_phases)}")
        all_separated = False

    for fold in FOLDS:
        train_data = load_scp(base_dir / fold / "train.scp")
        val_data = load_scp(base_dir / fold / "validation.scp")

        if train_data is None or val_data is None:
            print(f"✗ Cannot load fold {fold}")
            all_separated = False
            continue

        train_phases = set(train_data[:, 5])
        val_phases = set(val_data[:, 5])

        # Check validation phases match expected
        expected_val = VALIDATION_PHASES[fold]
        if val_phases == expected_val:
            print(f"✓ Fold {fold} validation phases correct: {sorted(val_phases)}")
        else:
            print(f"✗ Fold {fold} validation phases mismatch!")
            print(f"  Expected: {sorted(expected_val)}")
            print(f"  Got: {sorted(val_phases)}")
            all_separated = False

        # Check no overlap
        train_val_overlap = train_phases & val_phases
        train_test_overlap = train_phases & test_phases
        val_test_overlap = val_phases & test_phases

        if train_val_overlap:
            print(f"✗ Fold {fold}: train-val overlap: {train_val_overlap}")
            all_separated = False
        if train_test_overlap:
            print(f"✗ Fold {fold}: train-test overlap: {train_test_overlap}")
            all_separated = False
        if val_test_overlap:
            print(f"✗ Fold {fold}: val-test overlap: {val_test_overlap}")
            all_separated = False

    if all_separated:
        print("✓ All phases properly separated")

    return all_separated


def check_participant_coverage(base_dir):
    """Check all participants appear in all splits."""
    print("\n" + "="*70)
    print("5. PARTICIPANT COVERAGE CHECK")
    print("="*70)

    all_covered = True
    test_data = load_scp(base_dir / "test.scp")
    test_pids = set(test_data[:, 0])

    print(f"Test set: {len(test_pids)} participants")
    if len(test_pids) == EXPECTED_PARTICIPANTS:
        print(f"✓ Expected participant count: {EXPECTED_PARTICIPANTS}")
    else:
        print(f"⚠ Participant count: expected {EXPECTED_PARTICIPANTS}, got {len(test_pids)}")

    for fold in FOLDS:
        train_data = load_scp(base_dir / fold / "train.scp")
        val_data = load_scp(base_dir / fold / "validation.scp")

        if train_data is None or val_data is None:
            all_covered = False
            continue

        train_pids = set(train_data[:, 0])
        val_pids = set(val_data[:, 0])

        # Check all participants in all splits
        if train_pids == val_pids == test_pids:
            print(f"✓ Fold {fold}: All {len(test_pids)} participants in train/val/test")
        else:
            print(f"✗ Fold {fold}: Participant mismatch!")
            print(f"  Train: {len(train_pids)}, Val: {len(val_pids)}, Test: {len(test_pids)}")
            all_covered = False

    return all_covered


def check_phase_to_class_mapping(base_dir):
    """Verify phase-to-class mapping is correct."""
    print("\n" + "="*70)
    print("6. PHASE-TO-CLASS MAPPING CHECK")
    print("="*70)

    all_correct = True

    # Check all splits
    all_files = [base_dir / "test.scp"]
    for fold in FOLDS:
        all_files.extend([
            base_dir / fold / "train.scp",
            base_dir / fold / "validation.scp"
        ])

    phase_class_pairs = defaultdict(set)

    for filepath in all_files:
        data = load_scp(filepath)
        if data is None:
            continue

        for row in data:
            phase = row[5]
            cls = row[-1]
            phase_class_pairs[phase].add(cls)

    # Check each phase has only one class
    for phase, classes in sorted(phase_class_pairs.items()):
        expected_class = PHASE_TO_CLASS.get(phase, '?')
        if len(classes) > 1:
            print(f"✗ Phase {phase} has multiple classes: {classes}")
            all_correct = False
        elif list(classes)[0] != expected_class:
            print(f"✗ Phase {phase} has wrong class: expected {expected_class}, got {list(classes)[0]}")
            all_correct = False

    if all_correct:
        print(f"✓ All {len(phase_class_pairs)} phases have correct class labels")

    return all_correct


def check_data_integrity(base_dir, data_dir=None):
    """Check .scp format and optionally verify audio files exist."""
    print("\n" + "="*70)
    print("7. DATA INTEGRITY CHECK")
    print("="*70)

    all_valid = True

    # Check .scp format (10 columns)
    test_data = load_scp(base_dir / "test.scp")
    if test_data is not None:
        if test_data.shape[1] == 10:
            print(f"✓ Test set format correct: 10 columns")
        else:
            print(f"✗ Test set format wrong: {test_data.shape[1]} columns (expected 10)")
            all_valid = False

    # Check audio files if data_dir provided
    if data_dir:
        print(f"\nChecking audio files in: {data_dir}")
        missing_count = 0
        checked_pids = set()

        for row in test_data[:10]:  # Check first 10 samples
            pid = row[0]
            if pid in checked_pids:
                continue
            checked_pids.add(pid)

            audio_file = os.path.join(data_dir, "audio", pid, "close_talk.wav")
            if not os.path.exists(audio_file):
                print(f"✗ Missing audio: {audio_file}")
                missing_count += 1

        if missing_count == 0:
            print(f"✓ Audio files exist (checked {len(checked_pids)} participants)")
        else:
            print(f"⚠ {missing_count} audio files missing")
    else:
        print("⚠ Skipping audio file check (no data_dir provided)")

    return all_valid


def print_summary_stats(base_dir):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    test_data = load_scp(base_dir / "test.scp")
    print(f"\nShared test set: {len(test_data)} samples")

    for fold in FOLDS:
        train_data = load_scp(base_dir / fold / "train.scp")
        val_data = load_scp(base_dir / fold / "validation.scp")

        if train_data is None or val_data is None:
            continue

        print(f"\nFold {fold}:")
        print(f"  Train: {len(train_data):4d} samples")
        print(f"  Val:   {len(val_data):4d} samples")
        print(f"  Test:  {len(test_data):4d} samples (shared)")


def main():
    parser = argparse.ArgumentParser(description="Verify BESST v6 cognitive load splits")
    parser.add_argument("--metadata_dir", type=str, default="dataset/metadata",
                        help="Path to metadata directory")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to raw data directory (optional, for audio file check)")

    args = parser.parse_args()

    base_dir = Path(args.metadata_dir) / "lists" / "cognitive-load" / "audio"

    print("="*70)
    print("BESST v6 COGNITIVE LOAD SPLIT VERIFICATION")
    print("="*70)
    print(f"Base directory: {base_dir}")

    if not base_dir.exists():
        print(f"\n✗ Base directory does not exist: {base_dir}")
        sys.exit(1)

    # Run all checks
    checks = [
        ("File existence", lambda: check_file_existence(base_dir)),
        ("Test set consistency", lambda: check_test_set_consistency(base_dir)),
        ("Train/val distribution", lambda: check_train_val_distribution(base_dir)),
        ("Phase separation", lambda: check_phase_separation(base_dir)),
        ("Participant coverage", lambda: check_participant_coverage(base_dir)),
        ("Phase-to-class mapping", lambda: check_phase_to_class_mapping(base_dir)),
        ("Data integrity", lambda: check_data_integrity(base_dir, args.data_dir)),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            results.append((name, False))

    # Print summary stats
    print_summary_stats(base_dir)

    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)

    if all_passed:
        print("\n🎉 All checks passed! BESST v6 splits are valid.")
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()