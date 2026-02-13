#!/usr/bin/env python3
"""Migration script to add fullname field to existing corpora.

This script adds the full official legal title (fullname) to each corpus
in corpora.json. The fullname is used for legal citations in PDF export.

Usage:
    python scripts/migrate_corpus_fullname.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

# Official EU legal titles for existing corpora
# These are the official Danish titles from EUR-Lex
FULLNAME_MAPPING = {
    "ai-act": "Europa-Parlamentets og Rådets forordning (EU) 2024/1689 af 13. juni 2024 om harmoniserede regler for kunstig intelligens og om ændring af forordning (EF) nr. 300/2008, (EU) nr. 167/2013, (EU) nr. 168/2013, (EU) 2018/858, (EU) 2018/1139 og (EU) 2019/2144 og direktiv 2014/90/EU, (EU) 2016/797 og (EU) 2020/1828 (forordningen om kunstig intelligens)",
    "cyberrobusthed": "Europa-Parlamentets og Rådets forordning (EU) 2024/2847 af 23. oktober 2024 om horisontale cybersikkerhedskrav til produkter med digitale elementer og om ændring af forordning (EU) nr. 168/2013 og (EU) 2019/1020 og direktiv (EU) 2020/1828 (forordningen om cyberrobusthed)",
    "data-act": "Europa-Parlamentets og Rådets forordning (EU) 2023/2854 af 13. december 2023 om harmoniserede regler for rimelig adgang til og anvendelse af data og om ændring af forordning (EU) 2017/2394 og direktiv (EU) 2020/1828 (dataforordningen)",
    "dora": "Europa-Parlamentets og Rådets forordning (EU) 2022/2554 af 14. december 2022 om digital operationel modstandsdygtighed i den finansielle sektor og om ændring af forordning (EF) nr. 1060/2009, (EU) nr. 648/2012, (EU) nr. 600/2014, (EU) nr. 909/2014 og (EU) 2016/1011",
    "elan": "Europa-Parlamentets og Rådets forordning (EU) 2025/1272 af 6. maj 2025 om rammer for adgang til landbrugsdata (ELAN)",
    "gdpr": "Europa-Parlamentets og Rådets forordning (EU) 2016/679 af 27. april 2016 om beskyttelse af fysiske personer i forbindelse med behandling af personoplysninger og om fri udveksling af sådanne oplysninger og om ophævelse af direktiv 95/46/EF (generel forordning om databeskyttelse)",
    "nis2": "Europa-Parlamentets og Rådets direktiv (EU) 2022/2555 af 14. december 2022 om foranstaltninger til sikring af et højt fælles cybersikkerhedsniveau i hele Unionen, om ændring af forordning (EU) nr. 910/2014 og direktiv (EU) 2018/1972 og om ophævelse af direktiv (EU) 2016/1148 (NIS 2-direktivet)",
    "rearm-europe": "Europa-Parlamentets og Rådets forordning (EU) 2025/2653 af 19. december 2025 om fremme af forsvarsrelaterede investeringer og sikring af forsyningssikkerheden i Den Europæiske Union (REARM)",
    "reg-sandkasser": "Europa-Parlamentets og Rådets forordning (EU) 2025/1420 af 17. juli 2025 om reguleringsmæssige sandkasser til fremme af innovation i den digitale sektor",
}


def migrate_corpora(corpora_path: Path, dry_run: bool = False) -> None:
    """Add fullname field to all corpora.

    Args:
        corpora_path: Path to corpora.json
        dry_run: If True, only print what would be changed without writing
    """
    if not corpora_path.exists():
        print(f"Error: {corpora_path} does not exist")
        return

    # Load existing data
    with open(corpora_path, encoding="utf-8") as f:
        data = json.load(f)

    corpora = data.get("corpora", {})
    changes_made = 0

    for corpus_id, corpus_data in corpora.items():
        if "fullname" in corpus_data:
            print(f"  {corpus_id}: already has fullname, skipping")
            continue

        fullname = FULLNAME_MAPPING.get(corpus_id)
        if fullname:
            corpus_data["fullname"] = fullname
            changes_made += 1
            print(f"  {corpus_id}: added fullname")
        else:
            print(f"  {corpus_id}: WARNING - no fullname mapping found")

    if dry_run:
        print(f"\nDry run: {changes_made} changes would be made")
        return

    if changes_made == 0:
        print("\nNo changes needed")
        return

    # Create backup
    backup_dir = corpora_path.parent / "backup"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"corpora_{timestamp}.json"
    shutil.copy(corpora_path, backup_path)
    print(f"\nBackup created: {backup_path}")

    # Write updated data
    with open(corpora_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    print(f"\nMigration complete: {changes_made} corpora updated")


def main():
    parser = argparse.ArgumentParser(description="Add fullname to existing corpora")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without writing")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    corpora_path = project_root / "data" / "processed" / "corpora.json"

    print(f"Migrating {corpora_path}")
    print("=" * 60)
    migrate_corpora(corpora_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
