"""
scripts/train_embedding.py

Skeleton script to fine-tune a sentence-transformers embedding model on triples.
Requires sentence-transformers and torch.

Usage (example):
  python scripts\train_embedding.py --train data/finetune/triples.txt --model-name sentence-transformers/all-MiniLM-L6-v2 --out models/biomed-embed
"""
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers import evaluation
from torch.utils.data import DataLoader


def read_triples(path: Path):
    examples = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            a, b, c = line.strip().split('\t')
            examples.append(InputExample(texts=[a, b, c]))
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--model-name', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--out', default='models/finetuned-embed')
    p.add_argument('--epochs', type=int, default=1)
    args = p.parse_args()

    train_examples = read_triples(Path(args.train))
    model = SentenceTransformer(args.model_name)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        output_path=args.out
    )
    print('Saved model to', args.out)


if __name__ == '__main__':
    main()
