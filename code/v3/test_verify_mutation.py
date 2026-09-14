"""Mutation test for code/v3/verify_manuscript.py (H12).

1. Runs the verifier on the UNMUTATED manuscript and records which manifest entries fail
   (the *baseline*).  A clean manuscript has an empty baseline; the v5 manuscript has two
   documented arithmetic discrepancies (KNOWN_BASELINE) that the verifier is *supposed* to
   report, so the test requires baseline == KNOWN_BASELINE rather than baseline == {}.
2. Copies the .tex to a temporary file and mutates ONE number at a time — N entries drawn at
   random (seeded) from the manifest entries that verify OK, plus the two adversarial cases of
   the review: the tab:tuning cell 5.23 -> 99.99 (which code/v2 accepted) and 5.23 -> 8.31
   (a value that legitimately occurs elsewhere in the same table, so any global substring
   search still finds it).  Each mutation must make the verifier exit 1 AND report the mutated
   entry itself as DISCREPANCIA / NO ENCONTRADA.
3. Exit 0 only if every mutation was caught and the baseline is exactly the known one.

    python code/v3/test_verify_mutation.py [--n 12] [--seed 0] [--tex PATH] [--v2]
    (--v2 runs the same mutations through code/v2/verify_manuscript.py for comparison)
"""
from __future__ import annotations
import argparse, json, random, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_manuscript as vm  # noqa: E402

# v5 carried two arithmetic slips that the verifier reported (gap_dR2_A: 0.073 -> 0.074; grid_grp_range_d6plus:
# 0.2 K -> 0.3 K). Both are corrected in v3, so the expected baseline of failing ids is EMPTY.
KNOWN_BASELINE: set[str] = set()


def run_verifier(tex: Path, manifest: Path, script: Path = HERE / "verify_manuscript.py", report: Path | None = None):
    cmd = [sys.executable, str(script), "--tex", str(tex)]
    if script.parent.name == "v3": cmd += ["--manifest", str(manifest), "--quiet"]   # v2 only knows --tex
    if report: cmd += ["--report", str(report)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def failing_ids(report: Path) -> set[str]:
    rep = json.load(open(report))
    return {r["id"] for r in rep["results"] if r["status"] in ("DISCREPANCIA", "NO ENCONTRADA")}


def bump(val: str) -> str:
    """A plausible-looking wrong value: last digit +1 (9 -> 8), keeping sign and decimals."""
    digits = list(val)
    for i in range(len(digits) - 1, -1, -1):
        if digits[i].isdigit():
            digits[i] = "8" if digits[i] == "9" else str(int(digits[i]) + 1); break
    out = "".join(digits)
    return out if out != val else val + "1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", type=Path, default=vm.DEFAULT_TEX)
    ap.add_argument("--manifest", type=Path, default=vm.DEFAULT_MANIFEST)
    ap.add_argument("--n", type=int, default=12, help="random mutations (>= 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v2", action="store_true", help="also run code/v2/verify_manuscript.py on each mutant")
    ap.add_argument("--out", type=Path, default=None, help="write the mutation log as JSON here")
    args = ap.parse_args()
    assert args.n >= 10, "the task requires at least 10 mutations"

    tmp = Path(tempfile.mkdtemp(prefix="verify_mut_"))
    base_rep = tmp / "baseline.json"
    rc0, out0 = run_verifier(args.tex, args.manifest, report=base_rep)
    baseline = failing_ids(base_rep)
    rep = json.load(open(base_rep))
    ok_entries = [r for r in rep["results"] if r["status"] == "OK" and r.get("span")]
    text = vm.strip_comments(args.tex.read_text(encoding="utf-8"))
    original = args.tex.read_text(encoding="utf-8")
    assert len(text) == len(original)

    print(f"tex: {args.tex}\nbaseline (sin mutar): exit {rc0}, entradas que fallan = {sorted(baseline) or '{}'}")
    print(f"  baseline == KNOWN_BASELINE {sorted(KNOWN_BASELINE)}: {baseline == KNOWN_BASELINE}")

    rng = random.Random(args.seed)
    picks = rng.sample(ok_entries, args.n)
    by_id = {r["id"]: r for r in rep["results"]}
    plan = [(r, bump(r["tex"] if isinstance(r["tex"], str) else r["tex"][0])) for r in picks]
    # the two adversarial cases of the review (tab:tuning, Published configuration, Dataset A random MAE)
    h12 = by_id["tab_tuning_A_a_fixed_Fixed_published_r"]   # the tab:tuning cell 5.23 (published config, random arm) of the H12 review case
    plan += [(h12, "99.99"), (h12, "8.31")]

    log, caught = [], 0
    for k, (r, mutant) in enumerate(plan):
        a, b = r["span"][0]
        assert original[a:b] == (r["tex"] if isinstance(r["tex"], str) else r["tex"][0]), (r["id"], original[a:b])
        mutated = original[:a] + mutant + original[b:]
        mtex = tmp / f"mut_{k:02d}.tex"; mtex.write_text(mutated, encoding="utf-8")
        mrep = tmp / f"mut_{k:02d}.json"
        rc, _ = run_verifier(mtex, args.manifest, report=mrep)
        fails = failing_ids(mrep) - baseline
        hit = rc == 1 and r["id"] in fails
        caught += hit
        row = {"k": k, "id": r["id"], "where": r["where"][:90], "original": original[a:b], "mutant": mutant,
               "line": vm.line_of(text, a), "exit": rc, "new_failures": sorted(fails), "caught": hit}
        if args.v2:
            rc2, out2 = run_verifier(mtex, args.manifest, script=HERE.parent / "v2" / "verify_manuscript.py")
            m = re.search(r"comprobaciones OK: (\d+)\s+DISCREPANCIAS: (\d+)", out2)
            row["v2_exit"] = rc2; row["v2_summary"] = f"{m.group(1)} OK / {m.group(2)} discrepancias" if m else out2[-120:]
        log.append(row)
        print(f"  [{'✓' if hit else '✗'}] #{k:02d} {r['id']:40s} l.{row['line']:<4d} {original[a:b]!r} -> {mutant!r}  exit={rc}  "
              f"nuevas fallas={sorted(fails)}" + (f"  | v2: exit={row['v2_exit']} ({row['v2_summary']})" if args.v2 else ""))

    ok = caught == len(plan) and baseline == KNOWN_BASELINE
    print(f"\nmutaciones detectadas: {caught}/{len(plan)}   baseline conocida: {baseline == KNOWN_BASELINE}   -> {'PASS' if ok else 'FAIL'}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"tex": str(args.tex), "seed": args.seed, "n_random": args.n, "baseline_failing_ids": sorted(baseline),
                   "known_baseline": sorted(KNOWN_BASELINE), "mutations": log, "caught": caught, "total": len(plan), "pass": ok,
                   "procedencia": "code/v3/test_verify_mutation.py | mutant = last digit + 1 (or the two review cases 5.23->99.99 / 5.23->8.31); verifier = code/v3/verify_manuscript.py"},
                  open(args.out, "w"), indent=1, ensure_ascii=False)
        print(f"log: {args.out}")
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
