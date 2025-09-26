#!/usr/bin/env python3
"""
Install Argos Translate language packages for Turkish <-> English.
Usage (PowerShell):
  conda activate rag-med
  python scripts/install_argos.py
"""
from __future__ import annotations
import sys

try:
    import argostranslate.package as pkg
    import argostranslate.translate as tr
except Exception as e:
    print("[install_argos] argostranslate not installed. Install with: pip install argostranslate", file=sys.stderr)
    raise

def main():
    print("[install_argos] Updating package index...")
    pkg.update_package_index()
    print("[install_argos] Fetching available packages...")
    available = pkg.get_available_packages()
    wanted = [p for p in available if (p.from_code == 'en' and p.to_code == 'tr') or (p.from_code == 'tr' and p.to_code == 'en')]
    if not wanted:
        print("[install_argos] No en<->tr packages found in index.")
        return
    for p in wanted:
        print(f"[install_argos] Installing {p.from_code}->{p.to_code} ...")
        fp = p.download()
        pkg.install_from_path(fp)
    # smoke test
    try:
        s = tr.translate("Uykusuzluk yaygındır ve tedavi edilebilir.", "tr", "en")
        print("[install_argos] TR->EN sample:", s)
        s2 = tr.translate("Insomnia is common and treatable.", "en", "tr")
        print("[install_argos] EN->TR sample:", s2)
    except Exception as e:
        print("[install_argos] Smoke test failed:", e)

if __name__ == '__main__':
    main()
