"""Download and verify the FuelNet CatBoost model.

The model is stored as a GitHub Release asset because it is larger than
GitHub's normal 100 MiB file limit. This script keeps the repository small
while allowing a fresh clone to become fully runnable with one command.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
import tempfile
import urllib.error
import urllib.request

DEFAULT_URL = (
    "https://github.com/Nox-exe07/FUELNET/releases/download/v1.0-demo/catboost.cbm"
)
DEFAULT_DEST = pathlib.Path("catboost_v3_bundle") / "catboost.cbm"
DEFAULT_SHA_FILE = pathlib.Path("catboost_v3_bundle") / "model.sha256"


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sha(sha_file: pathlib.Path, explicit: str | None) -> str | None:
    if explicit:
        return explicit.strip().lower()
    if sha_file.exists():
        text = sha_file.read_text(encoding="utf-8").strip()
        if text:
            return text.split()[0].lower()
    return None


def download(url: str, dest: pathlib.Path, timeout: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=dest.name + ".", suffix=".part", dir=str(dest.parent)
    )
    os.close(tmp_fd)
    tmp_path = pathlib.Path(tmp_name)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "FuelNet-model-fetcher/1.0"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            next_report = 0
            with tmp_path.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        if pct >= next_report:
                            print(
                                f"  downloaded {done / (1024 * 1024):.1f} / "
                                f"{total / (1024 * 1024):.1f} MiB ({pct}%)",
                                flush=True,
                            )
                            next_report = pct + 5
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the FuelNet CatBoost model.")
    parser.add_argument(
        "--url",
        default=os.getenv("MODEL_URL", DEFAULT_URL),
        help="Release asset URL (default: MODEL_URL or the v1.0-demo asset).",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="Destination path for catboost.cbm.",
    )
    parser.add_argument(
        "--sha-file",
        default=str(DEFAULT_SHA_FILE),
        help="File containing the expected SHA-256.",
    )
    parser.add_argument(
        "--sha256",
        default=os.getenv("MODEL_SHA256"),
        help="Expected SHA-256 (overrides --sha-file).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a valid model already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Network timeout in seconds.",
    )
    args = parser.parse_args()

    dest = pathlib.Path(args.dest)
    sha_file = pathlib.Path(args.sha_file)
    expected = expected_sha(sha_file, args.sha256)

    if dest.exists() and dest.stat().st_size > 0 and not args.force:
        actual = sha256_of(dest)
        if expected is None:
            print(
                f"[FuelNet] Model already present at {dest} "
                f"(no checksum configured; use --force to replace)."
            )
            return 0
        if actual == expected:
            print(f"[FuelNet] Model already present and verified: {dest}")
            return 0
        print(
            f"[FuelNet] Existing model checksum does not match; re-downloading.\n"
            f"  expected {expected}\n  actual   {actual}"
        )

    print(f"[FuelNet] Downloading model from:\n  {args.url}")
    print(f"[FuelNet] Destination: {dest}")
    try:
        download(args.url, dest, args.timeout)
    except urllib.error.HTTPError as exc:
        print(
            f"[FuelNet] Download failed (HTTP {exc.code}).\n"
            "  Create the GitHub Release asset first, or download it manually:\n"
            f"  {args.url}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            f"[FuelNet] Download failed: {exc}\n"
            "  Check your internet connection and try again.\n"
            f"  Manual download: {args.url}",
            file=sys.stderr,
        )
        return 1

    actual = sha256_of(dest)
    if expected is not None and actual != expected:
        dest.unlink(missing_ok=True)
        print(
            "[FuelNet] Checksum verification FAILED; downloaded file removed.\n"
            f"  expected {expected}\n  actual   {actual}",
            file=sys.stderr,
        )
        return 1

    print(f"[FuelNet] Model ready: {dest} ({dest.stat().st_size / (1024 * 1024):.1f} MiB)")
    if expected is None:
        print("[FuelNet] WARNING: no checksum was configured for this download.")
    else:
        print(f"[FuelNet] SHA-256 verified: {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
