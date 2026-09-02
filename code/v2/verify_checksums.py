"""verify_checksums.py -- check data/*.csv|npz against data/CHECKSUMS.sha256.

Usage:  python code/v2/verify_checksums.py [--data DIR]
Exit 0 if every listed file exists and matches; 1 otherwise. Files that are absent are
reported as MISSING (Dataset A raw files must first be fetched with code/get_datasetA.py).
"""
from __future__ import annotations
import argparse, hashlib, os, sys
from pathlib import Path

ROOT = Path(os.environ.get("SUPERCON_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"


def sha256(p: Path, bs: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(bs), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA, help="data directory (default: ROOT/data)")
    a = ap.parse_args()
    manifest = a.data / "CHECKSUMS.sha256"
    if not manifest.exists():
        print(f"MISSING manifest {manifest}"); return 1
    bad = 0
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, name = line.split(None, 1)
        p = a.data / name.strip()
        if not p.exists():
            print(f"MISSING  {name}"); bad += 1; continue
        got = sha256(p)
        status = "OK      " if got == expected else "MISMATCH"
        bad += got != expected
        print(f"{status} {name}  {got}")
    print("all checksums OK" if bad == 0 else f"{bad} problem(s)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
