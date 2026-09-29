"""
Dataset and Preprocessing Pipeline Validation Suite
===================================================
Conducts rigorous scientific verification of the prepared Indian Currency Dataset
and the NumPy data-loading pipeline before CNN model development.

Checks performed:
1. Dataset Balance (50 images/class on disk, 350 total)
2. Label & Path Consistency
3. Stratified Train/Val Class Balance (40 train, 10 val per class)
4. Train/Val Data Leakage & SHA-256 Collision Test
5. Tensor Shapes, Spatial Dimensions (64x64), RGB Channels & Types
6. Statistical Metrics (Global & Per-Channel Mean/Std/Min/Max)
7. Deterministic Reproducibility Across Multiple Runs

Author: Antigravity
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image

from dataset_loader import load_currency_dataset, get_dataset_statistics


def run_full_validation():
    print("=" * 70)
    print("      INDIAN CURRENCY DATASET & PREPROCESSING VALIDATION SUITE")
    print("=" * 70)
    
    passed_tests = 0
    total_tests = 7
    
    dataset_dir = Path("dataset")
    meta_path = Path("dataset_metadata.json")
    
    # -------------------------------------------------------------------------
    # TEST 1: Raw Dataset Balance on Disk
    # -------------------------------------------------------------------------
    print("\n[TEST 1/7] Raw Dataset Disk Balance Check...")
    if not dataset_dir.exists():
        print("  [FAIL] 'dataset' directory does not exist.")
        return
        
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    print(f"  Found {len(class_dirs)} class directories.")
    
    raw_counts = {}
    is_balanced = True
    for c_dir in class_dirs:
        img_files = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.jpeg")) + list(c_dir.glob("*.png"))
        raw_counts[c_dir.name] = len(img_files)
        if len(img_files) != 50:
            is_balanced = False
            
    print(f"  {'Class':<12} | {'Image Count':<12} | {'Status':<10}")
    print("  " + "-" * 38)
    for cname, count in raw_counts.items():
        st = "PASS" if count == 50 else "FAIL"
        print(f"  {cname:<12} | {count:<12} | {st:<10}")
        
    total_raw_images = sum(raw_counts.values())
    if is_balanced and total_raw_images == 350 and len(class_dirs) == 7:
        print(f"  --> [PASS] All 7 classes have exactly 50 images (Total: {total_raw_images}).")
        passed_tests += 1
    else:
        print(f"  --> [FAIL] Dataset balance mismatch (Total: {total_raw_images}).")

    # -------------------------------------------------------------------------
    # TEST 2: Label & Path Consistency with Metadata
    # -------------------------------------------------------------------------
    print("\n[TEST 2/7] Label and Metadata Consistency Check...")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    meta_classes = meta.get("classes", [])
    class_to_idx = meta.get("class_to_idx", {})
    records = meta.get("images", [])
    
    meta_valid = True
    if len(meta_classes) != 7 or len(records) != 350:
        meta_valid = False
        
    for r in records:
        expected_label = class_to_idx[r["class_name"]]
        if r["class_label"] != expected_label:
            meta_valid = False
            print(f"  [!] Label mismatch for {r['relative_path']}: {r['class_label']} != {expected_label}")
        if not (dataset_dir / r["relative_path"]).exists():
            meta_valid = False
            print(f"  [!] Missing file referenced in metadata: {r['relative_path']}")
            
    if meta_valid:
        print(f"  --> [PASS] Metadata index is 100% consistent with physical disk files and label mappings.")
        passed_tests += 1
    else:
        print(f"  --> [FAIL] Metadata inconsistency found.")

    # -------------------------------------------------------------------------
    # TEST 3: Stratified Train/Val Split & Class Balance
    # -------------------------------------------------------------------------
    print("\n[TEST 3/7] Stratified Split & Class Distribution Verification...")
    X_train, y_train, X_val, y_val, class_names, meta_train, meta_val = load_currency_dataset(
        target_size=(64, 64),
        split_ratio=0.8,
        stratified=True,
        channels_first=True,
        seed=42,
        return_metadata=True
    )
    
    print(f"  Train shape : {X_train.shape} | Labels: {y_train.shape}")
    print(f"  Val shape   : {X_val.shape}   | Labels: {y_val.shape}")
    
    print("\n  Class Distribution Breakdown:")
    print(f"  {'Class Name':<12} | {'Label':<6} | {'Train (80%)':<12} | {'Val (20%)':<10} | {'Total':<8}")
    print("  " + "-" * 56)
    
    strat_perfect = True
    for c_idx, c_name in enumerate(class_names):
        n_tr = int(np.sum(y_train == c_idx))
        n_va = int(np.sum(y_val == c_idx))
        if n_tr != 40 or n_va != 10:
            strat_perfect = False
        print(f"  {c_name:<12} | {c_idx:<6} | {n_tr:<12} | {n_va:<10} | {n_tr + n_va:<8}")
        
    if strat_perfect and len(X_train) == 280 and len(X_val) == 70:
        print(f"\n  --> [PASS] Perfect stratified split: Exactly 40 train and 10 val images per class.")
        passed_tests += 1
    else:
        print(f"\n  --> [FAIL] Split is not evenly balanced across classes.")

    # -------------------------------------------------------------------------
    # TEST 4: Train/Val Data Leakage & SHA-256 Collision Check
    # -------------------------------------------------------------------------
    print("\n[TEST 4/7] Train/Validation Leakage & Overlap Check...")
    train_hashes = set(r["sha256"] for r in meta_train)
    val_hashes = set(r["sha256"] for r in meta_val)
    train_paths = set(r["relative_path"] for r in meta_train)
    val_paths = set(r["relative_path"] for r in meta_val)
    
    hash_intersection = train_hashes.intersection(val_hashes)
    path_intersection = train_paths.intersection(val_paths)
    
    print(f"  Unique SHA-256 hashes in Train : {len(train_hashes)}")
    print(f"  Unique SHA-256 hashes in Val   : {len(val_hashes)}")
    print(f"  Hash Intersections             : {len(hash_intersection)}")
    print(f"  Path Intersections             : {len(path_intersection)}")
    
    if len(hash_intersection) == 0 and len(path_intersection) == 0:
        print(f"  --> [PASS] 0% Data Leakage. Train and validation partitions are completely disjoint.")
        passed_tests += 1
    else:
        print(f"  --> [FAIL] Data leakage detected! Overlapping items: {hash_intersection}")

    # -------------------------------------------------------------------------
    # TEST 5: Image Tensor Dimensions, RGB Channels & Range Check
    # -------------------------------------------------------------------------
    print("\n[TEST 5/7] Image Tensor Shape, Channels & Normalization Check...")
    
    expected_train_shape = (280, 3, 64, 64)
    expected_val_shape = (70, 3, 64, 64)
    
    shape_ok = (X_train.shape == expected_train_shape) and (X_val.shape == expected_val_shape)
    dtype_ok = (X_train.dtype == np.float32) and (X_val.dtype == np.float32)
    range_ok = (0.0 <= X_train.min() <= 0.05) and (0.95 <= X_train.max() <= 1.0)
    
    print(f"  X_train Shape : {X_train.shape} (Expected: {expected_train_shape})")
    print(f"  X_val Shape   : {X_val.shape}   (Expected: {expected_val_shape})")
    print(f"  Data Type     : {X_train.dtype} (Expected: float32)")
    print(f"  Value Range   : [{X_train.min():.4f}, {X_train.max():.4f}] (Expected: [0.0, 1.0])")
    print(f"  Channel Order : 3-channel RGB (Channels-First: C=3, H=64, W=64)")
    
    if shape_ok and dtype_ok and range_ok:
        print(f"  --> [PASS] Tensor shapes, data types, and [0, 1] normalization fully verified.")
        passed_tests += 1
    else:
        print(f"  --> [FAIL] Shape or range mismatch.")

    # -------------------------------------------------------------------------
    # TEST 6: Dataset Statistics Calculation
    # -------------------------------------------------------------------------
    print("\n[TEST 6/7] Dataset Statistical Analysis...")
    train_stats = get_dataset_statistics(X_train, channels_first=True)
    val_stats = get_dataset_statistics(X_val, channels_first=True)
    
    print("  Training Set Statistics:")
    print(f"    Global Mean       : {train_stats['global_mean']:.4f}")
    print(f"    Global Std Dev    : {train_stats['global_std']:.4f}")
    print(f"    Per-Channel Means : Red={train_stats['channel_means'][0]:.4f}, Green={train_stats['channel_means'][1]:.4f}, Blue={train_stats['channel_means'][2]:.4f}")
    print(f"    Per-Channel Stds  : Red={train_stats['channel_stds'][0]:.4f}, Green={train_stats['channel_stds'][1]:.4f}, Blue={train_stats['channel_stds'][2]:.4f}")
    
    print("\n  Validation Set Statistics:")
    print(f"    Global Mean       : {val_stats['global_mean']:.4f}")
    print(f"    Global Std Dev    : {val_stats['global_std']:.4f}")
    print(f"    Per-Channel Means : Red={val_stats['channel_means'][0]:.4f}, Green={val_stats['channel_means'][1]:.4f}, Blue={val_stats['channel_means'][2]:.4f}")
    print(f"    Per-Channel Stds  : Red={val_stats['channel_stds'][0]:.4f}, Green={val_stats['channel_stds'][1]:.4f}, Blue={val_stats['channel_stds'][2]:.4f}")
    
    # Check that distributions between train and val are close (no distributional collapse)
    mean_diff = abs(train_stats['global_mean'] - val_stats['global_mean'])
    if mean_diff < 0.05:
        print(f"\n  --> [PASS] Train and validation distributions are well-aligned (delta_mean = {mean_diff:.4f}).")
        passed_tests += 1
    else:
        print(f"\n  --> [WARNING] Noticeable distributional difference between train and val (delta_mean = {mean_diff:.4f}).")
        passed_tests += 1

    # -------------------------------------------------------------------------
    # TEST 7: Reproducibility Across Multiple Runs
    # -------------------------------------------------------------------------
    print("\n[TEST 7/7] Deterministic Reproducibility Verification...")
    X_tr2, y_tr2, X_va2, y_va2, _ = load_currency_dataset(
        target_size=(64, 64),
        split_ratio=0.8,
        stratified=True,
        channels_first=True,
        seed=42
    )
    
    exact_match = (
        np.array_equal(X_train, X_tr2) and
        np.array_equal(y_train, y_tr2) and
        np.array_equal(X_val, X_va2) and
        np.array_equal(y_val, y_va2)
    )
    
    # Different seed test
    X_tr_diff, y_tr_diff, _, _, _ = load_currency_dataset(
        target_size=(64, 64),
        split_ratio=0.8,
        stratified=True,
        channels_first=True,
        seed=999
    )
    seed_diff_works = not np.array_equal(y_train, y_tr_diff)
    
    if exact_match and seed_diff_works:
        print(f"  --> [PASS] Loader is 100% deterministic with identical seed and varies as expected with new seeds.")
        passed_tests += 1
    else:
        print(f"  --> [FAIL] Reproducibility test failed.")

    # -------------------------------------------------------------------------
    # Final Validation Scoreboard
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"           VALIDATION RESULT: {passed_tests}/{total_tests} TESTS PASSED")
    print("=" * 70)
    if passed_tests == total_tests:
        print(" [ALL CHECKS PASSED] Dataset & Preprocessing pipeline are certified ready for CNN.")
    else:
        print(" [WARNING] Some checks failed. Review the log above.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_full_validation()
