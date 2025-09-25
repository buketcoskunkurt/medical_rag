"""
scripts/prepare_finetune.py

Prepare datasets for fine-tuning embedding and reranker models.
This is a scaffold: it can convert a labeled TSV/CSV of (query, positive, negative)
triples into files suitable for sentence-transformers training.
"""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--triples', required=True, help='CSV/TSV with query,positive,negative')
    p.add_argument('--out-dir', default='data/finetune')
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sep = '\t' if args.triples.endswith('.tsv') else ','
    with open(args.triples, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=sep)
        trip_out = out / 'triples.txt'
        with trip_out.open('w', encoding='utf-8') as fo:
            for row in reader:
                if len(row) < 3:
                    continue
                q, pos, neg = row[0].strip(), row[1].strip(), row[2].strip()
                fo.write('\t'.join([q, pos, neg]) + '\n')

    print('Wrote triples to', trip_out)


if __name__ == '__main__':
    main()
