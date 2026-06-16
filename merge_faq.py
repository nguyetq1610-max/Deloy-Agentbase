#!/usr/bin/env python3
"""
Merge data/faq_zalopay.csv into data/canned_responses.csv
Usage: python3 merge_faq.py
"""

import csv
import os
import shutil
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
MAIN_CSV = os.path.join(BASE, "data", "canned_responses.csv")
FAQ_CSV  = os.path.join(BASE, "data", "faq_zalopay.csv")
BACKUP   = MAIN_CSV + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"

if not os.path.exists(FAQ_CSV):
    print(f"ERROR: {FAQ_CSV} not found. Run scrape_zalopay_faq.py first.")
    exit(1)

# Load FAQ rows
with open(FAQ_CSV, encoding="utf-8") as f:
    faq_rows = list(csv.DictReader(f))

if not faq_rows:
    print("FAQ file is empty, nothing to merge.")
    exit(0)

print(f"FAQ rows to merge: {len(faq_rows)}")

# Load existing titles to avoid duplicates
existing_titles = set()
with open(MAIN_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        existing_titles.add(row.get("Title", "").strip())

new_rows = [r for r in faq_rows if r.get("Title", "").strip() not in existing_titles]
print(f"New (non-duplicate) rows: {len(new_rows)}")

if not new_rows:
    print("All FAQ rows already exist in canned_responses.csv. Nothing to do.")
    exit(0)

# Backup original
shutil.copy2(MAIN_CSV, BACKUP)
print(f"Backed up original to {BACKUP}")

# Read fieldnames from main CSV
with open(MAIN_CSV, encoding="utf-8") as f:
    fieldnames = csv.DictReader(f).fieldnames

# Append new rows
with open(MAIN_CSV, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writerows(new_rows)

print(f"\n✅  Merged {len(new_rows)} FAQ articles into {MAIN_CSV}")
print("Rebuild Docker to apply: docker-compose up --build -d")
