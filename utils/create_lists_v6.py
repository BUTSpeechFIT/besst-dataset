#!/usr/bin/env python3
"""
Create BESST v6 cognitive load splits with phase-based stratification.

Similar to CLSE v5, this creates:
- Single equalized test set (shared across all folds)
- 5 jack-knifed train/validation splits (variants a-e)
- Splits by rebus PHASE (task instance), not by participant
- All 79 participants appear in train, val, and test

Test set design (fixed, ~500 samples, balanced):
- Class 1 (low):    R2_1, R3_1  (~165 samples)
- Class 2 (medium): R2_3, R3_3  (~160 samples)
- Class 3 (high):   R4_5, R4_6  (~181 samples)

Cognitive load is based on cumulative rebus count:
- Class 1: Early instances (R1_0/1, R2_0/1, R3_0/1, R4_0/1, Debriefing)
- Class 2: Middle instances (R1_2, R2_2/3, R3_2/3, R4_2/3)
- Class 3: Late instances (R3_4, R4_4/5/6)

IMPORTANT - Class Imbalance:
Train sets have imbalanced distribution (~50% class 1, ~40% class 2, ~10% class 3)
due to structural scarcity: only 4 class-3 phases exist (R3_4, R4_4, R4_5, R4_6),
and 2/4 are allocated to test set for balance. This is handled via class weights
during training (weights: [0.67, 0.84, 3.22]), following CLSE v5 methodology.
"""

import numpy as np
from pathlib import Path
from collections import Counter
import argparse


# Phase-to-class mapping (based on cumulative cognitive load)
# NOTE: R0_* (learning phase) and Debriefing (different structure) are excluded
PHASE_TO_CLASS = {
    # Class 1 (low cognitive load)
    'R1_0': '1', 'R1_1': '1',
    'R2_0': '1', 'R2_1': '1',
    'R3_0': '1', 'R3_1': '1',
    'R4_0': '1', 'R4_1': '1',

    # Class 2 (medium cognitive load)
    'R1_2': '2',
    'R2_2': '2', 'R2_3': '2',
    'R3_2': '2', 'R3_3': '2',
    'R4_2': '2', 'R4_3': '2',

    # Class 3 (high cognitive load)
    'R3_4': '3',
    'R4_4': '3', 'R4_5': '3', 'R4_6': '3',
}

# Phases to exclude from experiment (learning phase and debriefing)
EXCLUDED_PHASES = {'R0_0', 'R0_1', 'Debriefing'}

# Test set phases (fixed, equalized across classes)
TEST_PHASES = {'R2_1', 'R3_1', 'R2_3', 'R3_3', 'R4_5', 'R4_6'}

# Validation phases per fold (jack-knifed)
VALIDATION_PHASES = {
    'a': {'R1_0', 'R2_2', 'R3_4'},
    'b': {'R1_1', 'R3_2', 'R4_4'},
    'c': {'R2_0', 'R4_2', 'R3_4'},  # R3_4 reused (limited class 3 phases)
    'd': {'R3_0', 'R4_3', 'R4_4'},  # R4_4 reused
    'e': {'R4_0', 'R1_2', 'R3_4'},  # R3_4 reused
}


def load_speech_segments(metadata_dir):
    """Load speech segments from metadata."""
    segments_file = Path(metadata_dir) / "speech-segments.csv"
    segments = np.loadtxt(segments_file, dtype=object, delimiter=";", skiprows=1)

    # Filter to Rebus tasks only (column 6 is phase_name)
    rebus_mask = segments[:, 6] == 'Rebus'
    rebus_segments = segments[rebus_mask]

    return rebus_segments


def load_participants(metadata_dir):
    """Load participants with complete multimodal data."""
    participants_file = Path(metadata_dir) / "participants.csv"
    participants = np.loadtxt(participants_file, dtype=object, delimiter=";", skiprows=1)

    # Filter to participants with all modalities available (columns 6+)
    multilingual_mask = np.all(participants[:, 6:].astype(int), axis=1)
    filtered_pids = set(participants[multilingual_mask, 0])

    return filtered_pids


def create_scp_entry(segment):
    """
    Create .scp file entry from segment.

    Format: pid;sid;start;end;text;phase;phase_name;subjective_load;rebus_score;cognitive_load_class

    Columns from speech-segments.csv:
    [0] pid, [1] sid, [2] start, [3] end, [4] text, [5] phase, [6] phase_name, [7] subjective_load, [8] rebus_score
    """
    pid, sid, start, end, text, phase, phase_name, subj_load, rebus_score = segment

    # Get cognitive load class from phase
    cognitive_class = PHASE_TO_CLASS.get(phase, '1')  # Default to class 1 if not in mapping

    # Build scp line
    scp_line = f"{pid};{sid};{start};{end};{text};{phase};{phase_name};{subj_load};{rebus_score};{cognitive_class}"

    return scp_line


def print_split_stats(segments, split_name):
    """Print statistics for a split."""
    phases = [s[5] for s in segments]
    classes = [PHASE_TO_CLASS.get(s[5], '1') for s in segments]
    pids = set(s[0] for s in segments)

    phase_counts = Counter(phases)
    class_counts = Counter(classes)

    print(f"\n{split_name}:")
    print(f"  Total: {len(segments)} samples, {len(pids)} participants")
    print(f"  Phases: {sorted(set(phases))}")
    print(f"  Classes:")
    for cls in ['1', '2', '3']:
        count = class_counts.get(cls, 0)
        pct = count / len(segments) * 100 if len(segments) > 0 else 0
        print(f"    Class {cls}: {count:4d} ({pct:5.1f}%)")


def create_splits(segments, valid_pids, output_dir):
    """Create train/val/test splits for all variants (a-e)."""

    # Filter to valid participants and exclude R0/Debriefing phases
    segments = np.array([s for s in segments
                        if s[0] in valid_pids and s[5] not in EXCLUDED_PHASES])

    print(f"Total segments: {len(segments)} from {len(valid_pids)} participants")
    print(f"Excluded phases: {EXCLUDED_PHASES}")

    # Create output directories
    base_dir = Path(output_dir) / "lists" / "cognitive-load" / "audio"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Separate test set by phase
    test_segments = np.array([s for s in segments if s[5] in TEST_PHASES])

    # Write single shared test set
    test_file = base_dir / "test.scp"
    with open(test_file, 'w') as f:
        for seg in test_segments:
            f.write(create_scp_entry(seg) + '\n')

    print_split_stats(test_segments, "Test set (shared)")

    # Create 5 different train/val splits (by phase)
    fold_names = ['a', 'b', 'c', 'd', 'e']

    for fold_name in fold_names:
        val_phases = VALIDATION_PHASES[fold_name]

        # Separate validation and train by phase
        val_segments = np.array([s for s in segments if s[5] in val_phases])
        train_segments = np.array([s for s in segments
                                   if s[5] not in TEST_PHASES and s[5] not in val_phases])

        # Create fold directory
        fold_dir = base_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Write train set
        train_file = fold_dir / "train.scp"
        with open(train_file, 'w') as f:
            for seg in train_segments:
                f.write(create_scp_entry(seg) + '\n')

        # Write validation set
        val_file = fold_dir / "validation.scp"
        with open(val_file, 'w') as f:
            for seg in val_segments:
                f.write(create_scp_entry(seg) + '\n')

        print(f"\n{'='*60}")
        print(f"FOLD {fold_name.upper()}")
        print(f"{'='*60}")
        print_split_stats(train_segments, "Train")
        print_split_stats(val_segments, "Validation")
        print(f"Test: {len(test_segments)} samples (shared across all folds)")


def main():
    parser = argparse.ArgumentParser(description="Create BESST v6 cognitive load splits")
    parser.add_argument("--metadata_dir", type=str, default="dataset/metadata",
                        help="Path to metadata directory")
    parser.add_argument("--output_dir", type=str, default="dataset/metadata",
                        help="Path to output directory")

    args = parser.parse_args()

    print("="*70)
    print("Creating BESST v6 Cognitive Load Splits (Phase-Based)")
    print("="*70)
    print(f"Metadata directory: {args.metadata_dir}")
    print(f"Output directory: {args.output_dir}")

    # Load data
    segments = load_speech_segments(args.metadata_dir)
    valid_pids = load_participants(args.metadata_dir)

    print(f"\nLoaded {len(segments)} Rebus segments from {len(valid_pids)} valid participants")

    # Create splits
    create_splits(segments, valid_pids, args.output_dir)

    print("\n" + "="*70)
    print("✓ BESST v6 splits created successfully!")
    print("="*70)
    print(f"Location: {Path(args.output_dir) / 'lists' / 'cognitive-load' / 'audio'}")
    print("\nSplit strategy:")
    print("  - Test: Fixed phases (R2_1, R3_1, R2_3, R3_3, R4_5, R4_6)")
    print("  - Validation: Different phases per fold (jack-knifed)")
    print("  - Train: Remaining phases")
    print("  - All 79 participants in all splits")
    print("\nNext steps:")
    print("  1. Install besst-dataset package: pip install -e .")
    print("  2. Run experiments with phase-based splits")
    print("  3. Compare with CLSE v5 results for transfer analysis")


if __name__ == "__main__":
    main()
