"""
Quick test script for Phase 2 — loads CSV, runs pipeline, prints shapes + leak check
"""
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import run_pipeline

if __name__ == "__main__":
    print("Running test_preprocess.py — loading CSV and running pipeline...")
    result = run_pipeline()
    
    print("\n=== FINAL OUTPUT SUMMARY ===")
    print(f"Train pool (2000-2020) numeric_scaled shape: {result['train_pool']['numeric_scaled'].shape}")
    print(f"Train pool ohe shape: {result['train_pool']['ohe'].shape}")
    print(f"Test (2021-2022) numeric_scaled shape: {result['test']['numeric_scaled'].shape}")
    print(f"Test ohe shape: {result['test']['ohe'].shape}")
    print(f"Train pool make vocab: {result['mappings']['make_vocab_size']}, model vocab: {result['mappings']['model_vocab_size']}")
    
    # Print GroupKFold split sizes already printed in run_pipeline, but re-summarize
    print(f"\nNumber of GroupKFold splits: {len(result['splits'])}")
    for i, (train_idx, val_idx) in enumerate(result['splits']):
        print(f"  Fold {i+1}: train {len(train_idx)}, val {len(val_idx)}")
    
    if result['leak_passed']:
        print("\nLEAK CHECK PASSED — No make+model overlap between train/val folds.")
    else:
        print("\nLEAK DETECTED — Fix required!")
        sys.exit(1)

    # Also test dataset class
    print("\n=== Testing PyTorch Dataset ===")
    from src.data.dataset import FuelConsumptionDataset
    train_ds = FuelConsumptionDataset.from_processed_pickle("train_pool")
    test_ds = FuelConsumptionDataset.from_processed_pickle("test")
    print(f"Dataset train_pool len: {len(train_ds)}, test len: {len(test_ds)}")
    num, cat, tgt = train_ds[0]
    print(f"Sample from train_ds[0]: numeric {num.shape}, cat make {cat['make'].item()}, model {cat['model'].item()}, targets {tgt.tolist()}")
    print(f"Dataset test passed — returns (numeric_tensor, categorical_dict, targets_tensor) as required.")
    
    print("\nAll checks passed.")
