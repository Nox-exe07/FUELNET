"""Assemble and upload the FuelNet Docker Space to Hugging Face.

The Space repository contains only code + small runtime artifacts + the prebuilt
frontend. The 100.8 MiB CatBoost model is downloaded from the GitHub Release
inside the Space's Docker build, so it never bloats the git repository.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deploy" / "hf-space"

BUNDLE_FILES = [
    "metadata.json",
    "te_mappings.json",
    "test_predictions.csv",
    "per_source.csv",
    "test_metrics.csv",
    "val_metrics.csv",
    "model.sha256",
]
PREPARED_FILES = [
    "feature_cols.json",
    "feature_median.pkl",
    "y_train_orig.npy",
]


def copy_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"required file missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: pathlib.Path, dst: pathlib.Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"required directory missing: {src}")
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache"
        ),
    )


def assemble(stage: pathlib.Path) -> None:
    copy_file(DEPLOY_DIR / "Dockerfile", stage / "Dockerfile")
    copy_file(DEPLOY_DIR / "README.md", stage / "README.md")
    copy_file(ROOT / "requirements-api.txt", stage / "requirements-api.txt")
    copy_file(ROOT / ".env.example", stage / ".env.example")
    copy_file(
        ROOT / "data" / "Fuel_Consumption_2000-2022.csv",
        stage / "data" / "Fuel_Consumption_2000-2022.csv",
    )

    copy_tree(ROOT / "src", stage / "src")
    copy_tree(ROOT / "configs", stage / "configs")
    copy_tree(ROOT / "scripts", stage / "scripts")
    copy_tree(ROOT / "frontend" / "dist", stage / "frontend" / "dist")

    for name in BUNDLE_FILES:
        copy_file(ROOT / "catboost_v3_bundle" / name, stage / "catboost_v3_bundle" / name)
    for name in PREPARED_FILES:
        copy_file(
            ROOT / "prepared_dataset" / "prepared_data" / name,
            stage / "prepared_dataset" / "prepared_data" / name,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy FuelNet to a Hugging Face Docker Space.")
    parser.add_argument("--space", default="FuelNet", help="Space repository name.")
    parser.add_argument("--user", default=None, help="Hugging Face username (default: token owner).")
    parser.add_argument("--private", action="store_true", help="Create a private Space.")
    parser.add_argument("--dry-run", action="store_true", help="Assemble files but do not upload.")
    args = parser.parse_args()

    user = args.user or "YOUR_USER"
    repo_id = f"{user}/{args.space}"

    with tempfile.TemporaryDirectory(prefix="fuelnet-hf-space-") as tmp:
        stage = pathlib.Path(tmp)
        print("[FuelNet] Assembling deployment folder...")
        assemble(stage)
        total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
        print(f"[FuelNet] Deployment folder size: {total / (1024 * 1024):.2f} MiB")
        print(f"[FuelNet] Target Space: {repo_id}")

        if args.dry_run:
            print(f"[FuelNet] Dry run complete. Files are in: {stage}")
            print("[FuelNet] (The temporary folder is removed when this script exits.)")
            return 0

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not token:
            print(
                "HF_TOKEN is not set. Create a write token at https://huggingface.co/settings/tokens\n"
                "then run:\n"
                "  set HF_TOKEN=hf_...\n"
                "  venv\\Scripts\\python.exe scripts\\deploy_hf_space.py",
                file=sys.stderr,
            )
            return 1

        try:
            from huggingface_hub import HfApi
        except ImportError:
            print(
                "huggingface_hub is not installed. Run:\n"
                "  venv\\Scripts\\python.exe -m pip install huggingface_hub",
                file=sys.stderr,
            )
            return 1

        api = HfApi(token=token)
        if not args.user:
            user = api.whoami(token=token)["name"]
            repo_id = f"{user}/{args.space}"
            print(f"[FuelNet] Resolved Space: {repo_id}")

        print("[FuelNet] Creating Space if needed...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
            token=token,
        )

        print("[FuelNet] Uploading files...")
        api.upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy FuelNet Docker Space",
            token=token,
        )

    page = f"https://huggingface.co/spaces/{repo_id}"
    app = f"https://{user.lower()}-{args.space.lower()}.hf.space"
    print(f"[FuelNet] Uploaded: {page}")
    print(f"[FuelNet] Expected app URL: {app}")
    print("[FuelNet] The Space may take a few minutes to build. Check the Space page for logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
