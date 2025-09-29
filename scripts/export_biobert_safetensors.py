#!/usr/bin/env python3
"""
Export BioBERT weights to safetensors locally to avoid torch.load restrictions in CUDA envs < 2.6.
Usage:
  python scripts/export_biobert_safetensors.py \
    --model dmis-lab/biobert-base-cased-v1.1 \
    --out   models/biobert-base-cased-v1.1-sf

Run this in a CPU-safe env (torch >= 2.6), e.g. the `rag-export` conda env.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dmis-lab/biobert-base-cased-v1.1")
    p.add_argument("--out", default="models/biobert-base-cased-v1.1-sf")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[export] downloading model: {args.model}")
    mdl = AutoModel.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)

    print("[export] converting to safetensors …")
    state = mdl.state_dict()
    # Ensure contiguous CPU tensors for reliability
    state = {k: v.contiguous().cpu() for k, v in state.items()}
    save_file(state, str(out_dir / "model.safetensors"))

    print("[export] saving config + tokenizer …")
    mdl.config.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    print(f"[export] done. saved to: {out_dir}")


if __name__ == "__main__":
    main()
