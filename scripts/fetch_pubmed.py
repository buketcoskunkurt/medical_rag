# scripts/fetch_pubmed.py
# PubMed'den makaleleri arayın ve indirin
import argparse, json, time, sys
from pathlib import Path
from Bio import Entrez
from datetime import datetime
import re


def parse_date_to_year(dstr):
    # try to extract a 4-digit year from common PubMed date formats
    if not dstr:
        return None
    m = re.search(r"(19|20)\d{2}", str(dstr))
    return int(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser(description="Fetch PubMed articles and save as JSONL compatible with build_index.py")
    ap.add_argument("--query", required=True, help="PubMed query string")
    ap.add_argument("--retmax", type=int, default=1000, help="Maximum number of ids to retrieve (esearch retmax)")
    ap.add_argument("--batch-size", type=int, default=50, help="How many ids to efetch per request")
    ap.add_argument("--email", default="example@example.com", help="Contact email for NCBI Entrez")
    ap.add_argument("--api_key", default=None, help="NCBI API key (optional) to increase rate limits")
    ap.add_argument("--out", default="data/pubmed.jsonl", help="Output jsonl file")
    ap.add_argument("--exclude", default=None, help="Comma-separated list of terms to exclude (case-insensitive). Example: autism,cancer")
    ap.add_argument("--mindate", type=int, default=None, help="Minimum publication year to include (YYYY)")
    ap.add_argument("--maxdate", type=int, default=None, help="Maximum publication year to include (YYYY)")
    ap.add_argument("--dedup", action="store_true", help="Remove duplicate PMIDs in output")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file instead of appending")
    args = ap.parse_args()

    Entrez.email = args.email
    if args.api_key:
        Entrez.api_key = args.api_key

    exclude_terms = [t.strip().lower() for t in (args.exclude or "").split(",") if t.strip()]
    exclude_pat = re.compile("|".join(re.escape(t) for t in exclude_terms), re.I) if exclude_terms else None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'w' if args.overwrite else 'a'

    print(f"[search] query={args.query} retmax={args.retmax}")
    s = Entrez.esearch(db="pubmed", term=args.query, retmax=args.retmax, sort="relevance")
    r = Entrez.read(s)
    s.close()
    ids = r.get("IdList", [])
    print(f"[found] {len(ids)} ids")

    written = 0
    skipped = 0
    duplicates = 0
    seen = set()

    # if not overwriting, gather existing ids to avoid duplicates
    if not args.overwrite and out_path.exists() and args.dedup:
        with out_path.open('r', encoding='utf-8') as rf:
            for line in rf:
                try:
                    obj = json.loads(line)
                    seen.add(str(obj.get('id')))
                except Exception:
                    continue

    batch = args.batch_size
    sleep_sec = 0.34 if not getattr(Entrez, 'api_key', None) else 0.12

    with out_path.open(mode, encoding='utf-8') as f:
        for start in range(0, len(ids), batch):
            chunk = ids[start:start+batch]
            if not chunk:
                break
            try:
                h = Entrez.efetch(db="pubmed", id=",".join(chunk), rettype="abstract", retmode="xml")
                papers = Entrez.read(h)
                h.close()
            except Exception as e:
                print(f"[efetch error] {e}")
                time.sleep(sleep_sec)
                continue

            for art in papers.get("PubmedArticle", []):
                pmid = str(art.get("MedlineCitation", {}).get("PMID", ""))
                if args.dedup and pmid in seen:
                    duplicates += 1
                    continue

                art_info = art.get("MedlineCitation", {}).get("Article", {})
                title = art_info.get("ArticleTitle", "")
                abstract = ""
                if art_info.get("Abstract") and art_info["Abstract"].get("AbstractText"):
                    abstract = " ".join(map(str, art_info["Abstract"]["AbstractText"]))

                pubdate = None
                # try several places for date info
                date_node = art.get("MedlineCitation", {}).get("Article", {}).get("Journal", {}).get("JournalIssue", {}).get("PubDate")
                if not date_node:
                    date_node = art.get("MedlineCitation", {}).get("DateCreated")
                pub_year = parse_date_to_year(date_node)

                # apply date filters if present
                if args.mindate and pub_year and pub_year < args.mindate:
                    skipped += 1
                    continue
                if args.maxdate and pub_year and pub_year > args.maxdate:
                    skipped += 1
                    continue

                text_blob = " ".join([str(title or ''), str(abstract or '')])
                if exclude_pat and exclude_pat.search(text_blob):
                    skipped += 1
                    continue

                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                if title or abstract:
                    obj = {"id": pmid, "title": str(title), "text": str(abstract), "url": url}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    written += 1
                    if args.dedup:
                        seen.add(pmid)

            time.sleep(sleep_sec)

    print(f"[write] written={written} skipped={skipped} duplicates_skipped={duplicates} -> {out_path}")


if __name__ == "__main__":
    main()
