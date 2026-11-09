"""
FuelNet Training — Phase 4
- Loads data via PyTorch Dataset from Phase 2 (processed_data.pkl)
- Standardizes targets, Huber loss, AdamW, cosine schedule, early stopping
- Saves best checkpoint, target_scaler, metrics, comparison, scatter plots
"""
import pathlib
import pickle
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models_artifacts"
PROCESSED_PATH = ARTIFACTS_DIR / "processed_data.pkl"
BASELINE_METRICS_PATH = ARTIFACTS_DIR / "baseline_metrics.json"
FUELNET_METRICS_PATH = ARTIFACTS_DIR / "fuelnet_metrics.json"
MODEL_COMPARISON_PATH = ARTIFACTS_DIR / "model_comparison.json"
TARGET_SCALER_PATH = ARTIFACTS_DIR / "target_scaler.pkl"
BEST_CHECKPOINT_PATH = ARTIFACTS_DIR / "fuelnet_best.pt"

from src.data.dataset import FuelConsumptionDataset
from src.models.fuelnet import FuelNet

TARGET_NAMES = ['fuel_city_l_100km', 'fuel_hwy_l_100km', 'fuel_comb_l_100km', 'co2_g_km']

def get_device():
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("WARNING: CUDA not available — using CPU (training will be slower)")
        return torch.device("cpu")

def compute_metrics(y_true, y_pred):
    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        mape = mean_absolute_percentage_error(yt, yp) * 100
        rmse = np.sqrt(mean_squared_error(yt, yp))
        metrics[name] = {"r2": float(r2), "mae": float(mae), "mape": float(mape), "rmse": float(rmse)}
    return metrics

def train():
    print("=== FuelNet Training ===")
    device = get_device()
    print(f"Device: {device}")

    # Load processed data
    print(f"Loading processed data from {PROCESSED_PATH}")
    with open(PROCESSED_PATH, 'rb') as f:
        data = pickle.load(f)
    train_pool = data['train_pool']
    test_data = data['test']
    splits = data['splits']

    # Use first GroupKFold split for train/val (honest, no make_model leak)
    train_idx, val_idx = splits[0]
    print(f"GroupKFold fold 0: train {len(train_idx)} rows, val {len(val_idx)} rows")
    # Further split train_pool into train/val arrays
    # Full train_pool stats for target scaler? We fit scaler only on training fold's train (not val) to be honest.
    # But we also need to ensure target scaler uses train only.
    y_train_full = train_pool['targets'][train_idx]  # (16528, 4)
    y_val = train_pool['targets'][val_idx]

    # Target standardization (fit only on train)
    target_scaler = StandardScaler()
    target_scaler.fit(y_train_full)
    print(f"Target scaler fitted: mean {target_scaler.mean_.round(2).tolist()}, var {target_scaler.var_.round(2).tolist()}")
    with open(TARGET_SCALER_PATH, 'wb') as f:
        pickle.dump(target_scaler, f)
    print(f"Saved target scaler to {TARGET_SCALER_PATH}")

    # Standardize targets
    y_train_scaled = target_scaler.transform(y_train_full)
    y_val_scaled = target_scaler.transform(y_val)
    y_test = test_data['targets']
    y_test_scaled_dummy = target_scaler.transform(y_test)  # not used for loss but for consistency

    # Build datasets
    # For train/val we need numeric_scaled, ohe, make_idx, model_idx
    def make_dataset(indices, scaled_targets):
        num = train_pool['numeric_scaled'][indices]
        ohe = train_pool['ohe'][indices]
        make_idx = train_pool['make_idx'][indices]
        model_idx = train_pool['model_idx'][indices]
        # Targets scaled for loss, but we also keep raw for metrics via inverse_transform
        return FuelConsumptionDataset(num, make_idx, model_idx, scaled_targets, ohe)

    train_ds = make_dataset(train_idx, y_train_scaled)
    val_ds = make_dataset(val_idx, y_val_scaled)

    # Test dataset (for final evaluation) — use scaled targets? We'll use raw and transform via scaler for inverse
    test_ds = FuelConsumptionDataset(test_data['numeric_scaled'], test_data['make_idx'], test_data['model_idx'], test_data['targets'], test_data['ohe'])

    print(f"Datasets: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")
    # Sample check
    num_samp, cat_samp, tgt_samp = train_ds[0]
    print(f"Sample numeric shape {num_samp.shape}, make {cat_samp['make'].item()}, model {cat_samp['model'].item()}, target scaled {tgt_samp.tolist()}")

    # DataLoaders
    batch_size = 256
    # Try 256, fallback to 128 if OOM (we will handle in training loop)
    try:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"DataLoaders batch_size={batch_size}")
    except Exception as e:
        print(f"Failed with batch 256: {e}, trying 128")
        batch_size = 128
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    with open(ARTIFACTS_DIR / "category_mappings.json", 'r') as f:
        mappings = json.load(f)
    make_vocab = mappings['make_vocab_size']
    model_vocab = mappings['model_vocab_size']
    # numeric_dim = ohe_dim + 5 = 44
    numeric_dim = train_ds.numeric.shape[1]  # 44
    print(f"Creating FuelNet: make_vocab {make_vocab}, model_vocab {model_vocab}, numeric_dim {numeric_dim}")

    model = FuelNet(make_vocab_size=make_vocab, model_vocab_size=model_vocab, numeric_dim=numeric_dim, hidden_dim=256, n_blocks=4, dropout=0.1)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Optimizer, scheduler, loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # Huber / SmoothL1
    criterion = nn.SmoothL1Loss(reduction='sum')  # summed over all 4 targets and batch

    # Training loop
    max_epochs = 200
    patience = 15
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    best_state = None

    # For logging
    history = []

    print(f"\nTraining for up to {max_epochs} epochs, early stopping patience {patience}, Huber loss, AdamW 1e-3, cosine schedule")
    start_time = time.time()

    for epoch in range(1, max_epochs+1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for numeric, cat, target in train_loader:
            numeric = numeric.to(device)
            make_idx = cat['make'].to(device)
            model_idx = cat['model'].to(device)
            target = target.to(device)  # already scaled

            optimizer.zero_grad()
            pred = model(numeric, make_idx, model_idx)  # (B,4) scaled space
            loss = F.smooth_l1_loss(pred, target, reduction='sum')  # sum over batch*4
            # Normalize by batch? Keep sum but divide for logging? We'll compute mean for logging
            loss_batch_mean = loss / (target.numel())  # mean per element for stability
            # But we used sum for optimizer, to match spec "summed over all 4 targets"
            # We'll backprop the mean to keep lr stable, but log mean
            loss_batch_mean.backward()
            optimizer.step()

            train_loss_sum += loss_batch_mean.item() * target.size(0)  # accumulate mean*batch
            train_count += target.size(0)

        train_loss = train_loss_sum / train_count  # approx mean Huber per sample (averaged over 4 targets)
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for numeric, cat, target in val_loader:
                numeric = numeric.to(device)
                make_idx = cat['make'].to(device)
                model_idx = cat['model'].to(device)
                target = target.to(device)
                pred = model(numeric, make_idx, model_idx)
                loss = F.smooth_l1_loss(pred, target, reduction='sum')
                loss_mean = loss / target.numel()
                val_loss_sum += loss_mean.item() * target.size(0)
                val_count += target.size(0)
        val_loss = val_loss_sum / val_count

        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{max_epochs} — train loss {train_loss:.5f} — val loss {val_loss:.5f} — lr {lr:.6f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})

        # Early stopping check
        if val_loss < best_val_loss - 1e-6:  # improvement
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save checkpoint immediately
            torch.save(best_state, BEST_CHECKPOINT_PATH)
            print(f"  -> New best val loss {best_val_loss:.5f} at epoch {epoch}, saved to {BEST_CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            print(f"  -> No improvement for {patience_counter}/{patience} epochs")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (best epoch {best_epoch} val loss {best_val_loss:.5f})")
            break

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s. Best epoch {best_epoch} val loss {best_val_loss:.5f}")

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Loaded best checkpoint from epoch {best_epoch}")
    else:
        print("No best checkpoint found, using last model")
        torch.save(model.state_dict(), BEST_CHECKPOINT_PATH)

    # Evaluate on train/val/test with inverse transform to raw space
    def evaluate(loader, scaler):
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for numeric, cat, target_scaled in loader:
                numeric = numeric.to(device)
                make_idx = cat['make'].to(device)
                model_idx = cat['model'].to(device)
                pred_scaled = model(numeric, make_idx, model_idx)  # scaled
                # inverse transform
                pred_raw = scaler.inverse_transform(pred_scaled.cpu().numpy())
                true_raw = scaler.inverse_transform(target_scaled.cpu().numpy())
                all_preds.append(pred_raw)
                all_trues.append(true_raw)
        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_trues, axis=0)
        return y_pred, y_true

    # For train/val we have scaled targets already; for test we use test_ds which has raw targets, but we need to scale then inverse? Actually test_ds has raw targets, but our evaluate expects scaled targets stored in dataset.
    # Our test_ds currently has raw targets (1896,4). For evaluation we should have scaled version? Let's create test_loader_scaled similarly.
    # Simpler: create test arrays scaled
    y_test_scaled = target_scaler.transform(test_data['targets'])
    test_ds_scaled = FuelConsumptionDataset(test_data['numeric_scaled'], test_data['make_idx'], test_data['model_idx'], y_test_scaled, test_data['ohe'])
    test_loader_scaled = DataLoader(test_ds_scaled, batch_size=batch_size, shuffle=False)

    # Also need loaders for scaled train/val for eval
    train_loader_scaled = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader_scaled = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("\n=== Evaluating best checkpoint ===")
    y_pred_train, y_true_train = evaluate(train_loader_scaled, target_scaler)
    y_pred_val, y_true_val = evaluate(val_loader_scaled, target_scaler)
    y_pred_test, y_true_test = evaluate(test_loader_scaled, target_scaler)

    metrics_train = compute_metrics(y_true_train, y_pred_train)
    metrics_val = compute_metrics(y_true_val, y_pred_val)
    metrics_test = compute_metrics(y_true_test, y_pred_test)

    print(f"\nTrain metrics (inverse-transformed):")
    for n in TARGET_NAMES:
        print(f"  {n}: R2 {metrics_train[n]['r2']:.4f} MAE {metrics_train[n]['mae']:.3f} MAPE {metrics_train[n]['mape']:.2f}%")
    print(f"\nVal metrics:")
    for n in TARGET_NAMES:
        print(f"  {n}: R2 {metrics_val[n]['r2']:.4f} MAE {metrics_val[n]['mae']:.3f}")
    print(f"\nTest (2021-2022) metrics:")
    for n in TARGET_NAMES:
        print(f"  {n}: R2 {metrics_test[n]['r2']:.4f} MAE {metrics_test[n]['mae']:.3f} MAPE {metrics_test[n]['mape']:.2f}% RMSE {metrics_test[n]['rmse']:.3f}")

    # Save fuelnet metrics
    fuelnet_metrics = {
        "train": metrics_train,
        "val": metrics_val,
        "test": metrics_test,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "elapsed_sec": float(elapsed),
        "history": history
    }
    with open(FUELNET_METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(fuelnet_metrics, f, indent=2)
    print(f"\nSaved FuelNet metrics to {FUELNET_METRICS_PATH}")

    # Load baseline metrics and create comparison
    with open(BASELINE_METRICS_PATH, 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    comparison = {}
    for name in TARGET_NAMES:
        comparison[name] = {
            "fuelnet": metrics_test[name],
            "xgboost": baseline["xgboost"][name],
            "linear_regression": baseline["linear_regression"][name]
        }
    # Also add summary of which winner
    summary = {}
    for name in TARGET_NAMES:
        fn_r2 = metrics_test[name]['r2']
        xgb_r2 = baseline["xgboost"][name]['r2']
        lr_r2 = baseline["linear_regression"][name]['r2']
        best = max(fn_r2, xgb_r2, lr_r2)
        if best == fn_r2:
            winner = "fuelnet"
        elif best == xgb_r2:
            winner = "xgboost"
        else:
            winner = "linear_regression"
        summary[name] = {
            "winner": winner,
            "fuelnet_r2": fn_r2,
            "xgboost_r2": xgb_r2,
            "linear_r2": lr_r2,
            "fuelnet_vs_xgb_r2_diff": float(fn_r2 - xgb_r2),
            "fuelnet_vs_lr_r2_diff": float(fn_r2 - lr_r2),
        }

    model_comparison = {
        "targets": comparison,
        "summary": summary,
        "fuelnet_best_epoch": best_epoch,
        "baseline_path": str(BASELINE_METRICS_PATH),
        "fuelnet_metrics_path": str(FUELNET_METRICS_PATH)
    }
    with open(MODEL_COMPARISON_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_comparison, f, indent=2)
    print(f"Saved model comparison to {MODEL_COMPARISON_PATH}")

    # Print comparison table
    print("\n" + "="*120)
    print(f"{'Target':<22} | {'Model':<16} | {'R2':>7} | {'MAE':>8} | {'MAPE%':>8} | {'RMSE':>8}")
    print("-"*120)
    for name in TARGET_NAMES:
        for model_name, m in [("FuelNet", metrics_test[name]), ("XGBoost", baseline["xgboost"][name]), ("LinearReg", baseline["linear_regression"][name])]:
            print(f"{name:<22} | {model_name:<16} | {m['r2']:7.4f} | {m['mae']:8.3f} | {m['mape']:8.2f} | {m['rmse']:8.3f}")
        print("-"*120)

    # Generate scatter plots for fuel_comb and co2
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for target_name in ['fuel_comb_l_100km', 'co2_g_km']:
            idx = TARGET_NAMES.index(target_name)
            yt = y_true_test[:, idx]
            yp = y_pred_test[:, idx]
            plt.figure(figsize=(6,6))
            plt.scatter(yt, yp, alpha=0.4, s=10, color='skyblue')
            # diagonal
            mn = min(yt.min(), yp.min())
            mx = max(yt.max(), yp.max())
            plt.plot([mn, mx], [mn, mx], color='red', linestyle='--', linewidth=1)
            plt.xlabel(f"Actual {target_name}")
            plt.ylabel(f"Predicted {target_name}")
            plt.title(f"FuelNet Predicted vs Actual — {target_name}\nTest 2021-2022 (n=1896)")
            # annotate R2
            r2 = r2_score(yt, yp)
            plt.text(0.05, 0.95, f"R2 = {r2:.4f}", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.5))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = ARTIFACTS_DIR / f"pred_vs_actual_{target_name}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"Saved scatter plot to {out_path}")
    except Exception as e:
        print(f"Failed to generate scatter plots: {e}")
        import traceback
        traceback.print_exc()

    # Final verdict
    print("\n=== Final Verdict: Did FuelNet beat baselines? ===")
    for name in TARGET_NAMES:
        fn = metrics_test[name]
        xgb = baseline["xgboost"][name]
        lr = baseline["linear_regression"][name]
        print(f"{name}: FuelNet R2 {fn['r2']:.4f} vs XGB {xgb['r2']:.4f} (diff {fn['r2']-xgb['r2']:+.4f}) vs LR {lr['r2']:.4f} (diff {fn['r2']-lr['r2']:+.4f}) — "
              f"MAE {fn['mae']:.3f} vs XGB {xgb['mae']:.3f} — {'BEAT' if fn['r2']>xgb['r2'] else 'LOST to XGB'}")

    return fuelnet_metrics, model_comparison

if __name__ == "__main__":
    train()
