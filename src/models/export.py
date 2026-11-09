"""
FuelNet Export — Phase 5
- Loads fuelnet_best.pt and saves TorchScript to fuelnet_ts.pt
- Writes model_card.json
"""
import pathlib
import pickle
import json
import datetime
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
BEST_CHECKPOINT_PATH = ARTIFACTS_DIR / "fuelnet_best.pt"
TS_PATH = ARTIFACTS_DIR / "fuelnet_ts.pt"
MODEL_CARD_PATH = ARTIFACTS_DIR / "model_card.json"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"
CATEGORY_MAPPINGS_PATH = ARTIFACTS_DIR / "category_mappings.json"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"

from src.models.fuelnet import FuelNet

def export_torchscript():
    print("=== Exporting FuelNet to TorchScript ===")
    # Load mappings
    with open(CATEGORY_MAPPINGS_PATH, 'r') as f:
        mappings = json.load(f)
    make_vocab = mappings['make_vocab_size']
    model_vocab = mappings['model_vocab_size']

    # Load preprocessor to get numeric dim
    with open(PREPROCESSOR_PATH, 'rb') as f:
        prep = pickle.load(f)
    numeric_dim = len(prep['numeric_cols']) + len(prep['ohe_feature_names'])  # 44
    print(f"Make vocab {make_vocab}, model vocab {model_vocab}, numeric_dim {numeric_dim}")

    # Create model and load weights
    model = FuelNet(make_vocab_size=make_vocab, model_vocab_size=model_vocab, numeric_dim=numeric_dim)
    state = torch.load(BEST_CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint from {BEST_CHECKPOINT_PATH}")

    # Example inputs for tracing: numeric (44), make_idx, model_idx
    example_numeric = torch.randn(1, numeric_dim)
    example_make = torch.tensor([0], dtype=torch.long)
    example_model = torch.tensor([0], dtype=torch.long)

    # TorchScript via tracing (more robust than scripting for this architecture)
    try:
        # Use scripting if possible, else trace
        scripted = torch.jit.trace(model, (example_numeric, example_make, example_model))
        # Test that scripted works
        with torch.no_grad():
            out = scripted(example_numeric, example_make, example_model)
        print(f"TorchScript traced successfully, output shape {out.shape}")
        scripted.save(str(TS_PATH))
        print(f"Saved TorchScript to {TS_PATH}")
    except Exception as e:
        print(f"TorchScript trace failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: save state_dict as ts? but spec wants ts.pt
        # Try scripting
        try:
            scripted = torch.jit.script(model)
            scripted.save(str(TS_PATH))
            print(f"Saved via scripting to {TS_PATH}")
        except Exception as e2:
            print(f"Scripting also failed: {e2}")
            raise

    # Verify file exists
    if TS_PATH.exists():
        print(f"TorchScript file size: {TS_PATH.stat().st_size} bytes")
    else:
        raise FileNotFoundError(f"Failed to create {TS_PATH}")

def write_model_card():
    print("\n=== Writing Model Card ===")
    # Load comparison metrics
    with open(MODEL_COMPARISON_PATH, 'r') as f:
        comparison = json.load(f)
    # Extract test metrics for fuelnet
    test_metrics = {}
    for target, vals in comparison['targets'].items():
        test_metrics[target] = vals['fuelnet']

    # Training date
    training_date = datetime.datetime.now().isoformat()

    model_card = {
        "model_version": "fuelnet-t-v1.0.0",
        "training_date": training_date,
        "dataset": {
            "source": "Kaggle ahmettyilmazz/fuel-consumption",
            "description": "Environment and Climate Change Canada / NRCan Fuel Consumption Ratings 2000-2022",
            "n_rows": 22556,
            "n_features": "numeric 5 + ohe 39 + make/model embeddings",
            "temporal_test": "2021-2022 held out (1896 rows)",
            "group_kfold": "5 splits grouped by make+model (2000-2020)"
        },
        "architecture": {
            "make_embedding_dim": 32,
            "model_embedding_dim": 64,
            "fusion_dim": 256,
            "residual_blocks": 4,
            "dropout": 0.1,
            "total_params": 776356
        },
        "training": {
            "loss": "Huber (smooth_l1) summed over 4 targets",
            "optimizer": "AdamW lr 1e-3 weight_decay 1e-4 cosine schedule",
            "batch_size": 256,
            "epochs": "up to 200, early stopping patience 15, best epoch 76",
            "target_scaling": "StandardScaler fitted on train, saved to target_scaler.pkl"
        },
        "test_metrics": test_metrics,
        "comparison_vs_baselines": comparison['summary'],
        "artifacts": {
            "fuelnet_best.pt": str(BEST_CHECKPOINT_PATH),
            "fuelnet_ts.pt": str(TS_PATH),
            "preprocessor.pkl": str(PREPROCESSOR_PATH),
            "category_mappings.json": str(CATEGORY_MAPPINGS_PATH),
            "target_scaler.pkl": str(ARTIFACTS_DIR / "target_scaler.pkl")
        },
        "known_limitations": [
            "Unseen makes/models at inference use 'unknown' embedding (index at end of vocab) — predictions are less certain, flagged via unseen_vehicle",
            "Predictions are for Canadian NRCan laboratory test cycle conditions (controlled dyno), not real-world driving",
            "Fuel types D/E/N/X/Z only — other fuels may not be accurate",
            "Temporal test 2021-2022 — performance may degrade for future years with new drivetrains (hybrid/electric shifts)",
            "Training data 2000-2022 passenger vehicles only — not valid for heavy trucks outside dataset distribution"
        ],
        "intended_use": "Research / eco-driving education + 2D simulator. Not for regulatory compliance.",
        "license": "MIT"
    }

    with open(MODEL_CARD_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_card, f, indent=2)
    print(f"Saved model card to {MODEL_CARD_PATH}")
    print(json.dumps(model_card, indent=2)[:2000])

if __name__ == "__main__":
    export_torchscript()
    write_model_card()
    print("\nExport complete.")
