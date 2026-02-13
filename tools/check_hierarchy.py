#!/usr/bin/env python3
"""Check hierarchical metadata in chunk files."""
import json
from pathlib import Path

def check_file(path: Path):
    with_article = 0
    with_chapter = 0
    with_section = 0
    with_recital = 0
    with_annex = 0
    no_hierarchy = 0
    total = 0

    with open(path) as f:
        for line in f:
            m = json.loads(line)['metadata']
            total += 1
            
            has_any = False
            if m.get('article'): with_article += 1; has_any = True
            if m.get('chapter'): with_chapter += 1; has_any = True
            if m.get('section'): with_section += 1; has_any = True
            if m.get('recital'): with_recital += 1; has_any = True
            if m.get('annex'): with_annex += 1; has_any = True
            
            if not has_any:
                no_hierarchy += 1

    print(f"{path.name}:")
    print(f"  Total: {total}")
    print(f"  article: {with_article}, chapter: {with_chapter}, section: {with_section}")
    print(f"  recital: {with_recital}, annex: {with_annex}")
    print(f"  NO hierarchy: {no_hierarchy}")
    print()

if __name__ == "__main__":
    for f in Path("data/processed").glob("*_chunks.jsonl"):
        check_file(f)
