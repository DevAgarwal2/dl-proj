#!/usr/bin/env python3
"""Fetch open-access PubMed abstracts for the 14 NIH findings.

Uses the public NCBI E-utilities API (no key required; <=3 requests/second).
Writes one JSONL file per finding into data/corpus/pubmed/.

Usage:
    python scripts/fetch_pubmed.py --per-finding 40 --out data/corpus/pubmed
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

FINDINGS = {
    "Atelectasis": "atelectasis[Title/Abstract] AND (chest radiograph[Title/Abstract] OR chest x-ray[Title/Abstract])",
    "Cardiomegaly": "cardiomegaly[Title/Abstract] AND (chest radiograph[Title/Abstract] OR chest x-ray[Title/Abstract])",
    "Effusion": "pleural effusion[Title/Abstract] AND (chest radiograph[Title/Abstract] OR chest x-ray[Title/Abstract])",
    "Infiltration": "pulmonary infiltrate[Title/Abstract] AND chest radiograph[Title/Abstract]",
    "Mass": "pulmonary mass[Title/Abstract] AND chest radiograph[Title/Abstract]",
    "Nodule": "pulmonary nodule[Title/Abstract] AND chest radiograph[Title/Abstract]",
    "Pneumonia": "pneumonia[Title/Abstract] AND (chest radiograph[Title/Abstract] OR chest x-ray[Title/Abstract])",
    "Pneumothorax": "pneumothorax[Title/Abstract] AND (chest radiograph[Title/Abstract] OR chest x-ray[Title/Abstract])",
    "Consolidation": "pulmonary consolidation[Title/Abstract] AND chest radiograph[Title/Abstract]",
    "Edema": "pulmonary edema[Title/Abstract] AND chest radiograph[Title/Abstract]",
    "Emphysema": "emphysema[Title/Abstract] AND (chest radiograph[Title/Abstract] OR computed tomography[Title/Abstract])",
    "Fibrosis": "pulmonary fibrosis[Title/Abstract] AND chest radiograph[Title/Abstract]",
    "Pleural_Thickening": "pleural thickening[Title/Abstract] AND chest radiograph[Title/Abstract]",
    "Hernia": "diaphragmatic hernia[Title/Abstract] AND chest radiograph[Title/Abstract]",
}


def _get(url: str, params: dict, retries: int = 4) -> bytes:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}", headers={"User-Agent": "OpenMed/0.1 (research project)"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt + 1} after {exc}")
            time.sleep(2 * (attempt + 1))
    return b""


def search_pmids(term: str, retmax: int) -> list[str]:
    payload = _get(
        f"{EUTILS}/esearch.fcgi",
        {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json", "sort": "relevance"},
    )
    data = json.loads(payload)
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    payload = _get(
        f"{EUTILS}/efetch.fcgi",
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
    )
    root = ET.fromstring(payload)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = "".join(article.find(".//ArticleTitle").itertext()) if article.find(".//ArticleTitle") is not None else ""
        abstract_parts = [
            "".join(node.itertext())
            for node in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(part.strip() for part in abstract_parts if part and part.strip())
        journal = article.findtext(".//Journal/Title") or ""
        year = article.findtext(".//JournalIssue/PubDate/Year") or article.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
        if not abstract:
            continue
        records.append(
            {
                "pmid": pmid,
                "title": title.strip(),
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-finding", type=int, default=40)
    parser.add_argument("--out", default="data/corpus/pubmed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for finding, term in FINDINGS.items():
        print(f"[{finding}] searching...")
        pmids = search_pmids(term, args.per_finding)
        time.sleep(0.4)  # respect the 3 req/s limit
        records = fetch_abstracts(pmids)
        time.sleep(0.4)
        path = out_dir / f"{finding}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                record["finding"] = finding
                record["text"] = f"{record['title']}\n\n{record['abstract']}"
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  wrote {len(records)} abstracts -> {path}")
        total += len(records)
    print(f"Done. {total} abstracts across {len(FINDINGS)} findings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
