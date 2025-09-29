#!/usr/bin/env python3
"""
Rewrite an answers CSV in-place, normalizing the Answer column:
- If Answer is the Turkish token "yetersiz" (case/spacing/punct variations allowed), replace with "insufficient".

Usage:
  python scripts/fix_answers_csv.py --file data/test/answers_10_for_manual_eval.csv
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def normalize_token(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # keep only ASCII letters for robust compare
    letters = ''.join(ch for ch in s.lower() if 'a' <= ch <= 'z')
    return letters


def fix_file(path: Path) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    replaced = 0
    with path.open("r", encoding="utf-8") as fh_in, tmp.open("w", newline="", encoding="utf-8") as fh_out:
        reader = csv.DictReader(fh_in)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise SystemExit("No columns found in CSV")
        writer = csv.DictWriter(fh_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            ans = (row.get("Answer") or "").strip()
            norm = normalize_token(ans)
            al = ans.lower()
            if norm == "yetersiz" or al == "yetersiz" or "yetersiz" in al or "yeter" in al:
                row["Answer"] = "insufficient"
                replaced += 1
            writer.writerow(row)
    tmp.replace(path)
    return replaced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to the answers CSV to rewrite in-place")
    args = ap.parse_args()
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    n = fix_file(path)
    print(f"Rewrote: {path} (replaced {n} entries)")


if __name__ == "__main__":
    main()
