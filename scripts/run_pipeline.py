#!/usr/bin/env python3
"""Orchestrator for the data pipeline: fetch, reduce, sample, and rebuild index.

Usage examples:
  # sample 200 per-file from existing data/raw
  python scripts/run_pipeline.py --sample --per-file 200 --out data/combined_random_perfile.jsonl

  # fetch per-topic (uses fetch scripts), then sample
  python scripts/run_pipeline.py --fetch --per-topic 200 --sample --out data/combined_random_perfile.jsonl

  # full pipeline and rebuild index
  python scripts/run_pipeline.py --fetch --per-topic 200 --sample --out data/combined_random_perfile.jsonl --rebuild-index --index-out data/index.faiss
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_fetch_all():
    # Legacy: batch fetch scripts removed/refactored. Skip fetch by default.
    print('Fetch step is not configured in this repository (batch fetch scripts not present). Skipping fetch.')


def run_reduce_sources():
    if Path('scripts/reduce_sources.py').exists():
        subprocess.check_call([sys.executable, 'scripts/reduce_sources.py', '--in', 'data/openalex.jsonl', '--out', 'data/openalex.jsonl'])
        subprocess.check_call([sys.executable, 'scripts/reduce_sources.py', '--in', 'data/pubmed.jsonl', '--out', 'data/pubmed.jsonl'])
    else:
        print('No scripts/reduce_sources.py found; skipping reduce')


def run_sample(per_file, outpath, seed=42):
    if Path('scripts/sample_per_file.py').exists():
        subprocess.check_call([sys.executable, 'scripts/sample_per_file.py', '--per-file', str(per_file), '--out', outpath, '--seed', str(seed)])
    else:
        print('No scripts/sample_per_file.py found; cannot sample')


def run_merge_combined(per_topic=200, out='data/combined_200.jsonl'):
    # Use prepare_combined.py if present, otherwise try merge_pubmed_openalex.py (legacy)
    if Path('scripts/prepare_combined.py').exists():
        subprocess.check_call([sys.executable, 'scripts/prepare_combined.py', '--per-topic', str(per_topic), '--out', out])
    elif Path('scripts/merge_pubmed_openalex.py').exists():
        print('prepare_combined.py not found; running legacy merge_pubmed_openalex.py')
        subprocess.check_call([sys.executable, 'scripts/merge_pubmed_openalex.py'])
    else:
        print('No merge script found (prepare_combined.py or merge_pubmed_openalex.py). Skipping merge.')


def run_index_rebuild(inpath, index_out, chunk_size=300, chunk_overlap=50):
    if Path('scripts/build_index.py').exists():
        subprocess.check_call([sys.executable, 'scripts/build_index.py', '--in', inpath, '--out', index_out, '--chunk_size', str(chunk_size), '--chunk_overlap', str(chunk_overlap)])
    else:
        print('No scripts/build_index.py found; skipping index rebuild')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--merge', action='store_true', help='Run merge/prepare combined step')
    p.add_argument('--reduce', action='store_true', help='Run reduce_sources on merged files')
    p.add_argument('--sample', action='store_true', help='Sample per-file usable records')
    p.add_argument('--per-topic', type=int, default=200, help='Records per topic for merge (used by prepare_combined)')
    p.add_argument('--per-file', type=int, default=200, help='Records per file for sampling')
    p.add_argument('--out', default='data/combined_random_perfile.jsonl', help='Output combined path for sampling')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rebuild-index', action='store_true', help='Rebuild FAISS index from input combined file')
    p.add_argument('--index-out', default='data/index.faiss', help='Index output path')
    p.add_argument('--in', dest='inpath', default='data/combined_random_perfile.jsonl', help='Input combined file for index rebuild')
    args = p.parse_args()

    if args.merge:
        print('Running merge/prepare combined step...')
        run_merge_combined(per_topic=args.per_topic, out=args.out)

    if args.reduce:
        print('Running reduce step...')
        run_reduce_sources()

    if args.sample:
        print(f'Sampling up to {args.per_file} usable records per file...')
        run_sample(args.per_file, args.out, seed=args.seed)

    if args.rebuild_index:
        print('Rebuilding index...')
        run_index_rebuild(args.inpath, args.index_out)

    print('Pipeline finished')


if __name__ == '__main__':
    main()
