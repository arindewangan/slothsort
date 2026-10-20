"""Build the SIMULATED file-history training dataset for SlothSort.

Generates 6,000 synthetic file records with a seeded RNG. Each record holds
file-metadata features (the only thing SlothSort ever sees — never file
contents) plus a label: KEEP / ARCHIVE / DELETE.

The labels follow common-sense "digital hoarding" heuristics with ~8% label
noise, so the classifier has to generalize rather than memorize a rule list.
This is SIMULATED data — no real user's Downloads folder was scanned.

Run:  python data/build_dataset.py
Out:  data/file_history.csv
"""
import csv
import os
import re

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_history.csv")
SEED = 42
N_ROWS = 6000
NOISE_RATE = 0.08

# Extension -> category id: 0 installer, 1 archive, 2 media, 3 document, 4 other
EXT_MAP = {
    ".exe": 0, ".msi": 0, ".dmg": 0, ".pkg": 0, ".deb": 0, ".apk": 0,
    ".zip": 1, ".tar": 1, ".gz": 1, ".rar": 1, ".7z": 1,
    ".mp4": 2, ".mkv": 2, ".mov": 2, ".jpg": 2, ".jpeg": 2, ".png": 2,
    ".heic": 2, ".mp3": 2, ".wav": 2, ".flac": 2,
    ".pdf": 3, ".docx": 3, ".xlsx": 3, ".pptx": 3, ".txt": 3, ".md": 3,
    ".csv": 3, ".epub": 3,
}

CLUTTER_PATTERNS = [
    r"final", r"copy", r"\(\d+\)", r"\bv\d+\b", r"backup", r"\bold\b", r"setup",
    r"installer", r"temp", r"untitled", r"screenshot", r"download",
]

KEEP, ARCHIVE, DELETE = 0, 1, 2

HEADER = ["size_mb_log", "age_days", "days_since_access", "ext_category",
          "name_clutter", "is_hidden_temp", "label"]

NAME_POOLS = {
    0: ["setup-chrome.exe", "installer-node.msi", "vlc-setup.dmg", "app-v2.pkg"],
    1: ["project-backup.zip", "photos-2023.tar.gz", "archive-old.rar"],
    2: ["trip-video.mp4", "IMG_2041.jpg", "song-final.mp3", "Screenshot 2024-01-02.png"],
    3: ["report-final.pdf", "budget-v2.xlsx", "notes.txt", "slides-final-final.pptx"],
    4: ["data.bin", "script.sh", "config.yaml", "notes(1).txt"],
}


def clutter_score(name):
    name = name.lower()
    hits = sum(1 for p in CLUTTER_PATTERNS if re.search(p, name))
    return min(1.0, hits / 2.0)


def label_row(size_mb, age_days, access_days, ext_cat, clutter, hidden):
    # Obvious junk first.
    if hidden:
        return DELETE
    if ext_cat == 0 and access_days > 180:
        return DELETE  # stale installer — re-downloadable any time
    if clutter > 0.6 and age_days > 120 and access_days > 120:
        return DELETE  # duplicate-ish clutter nobody touches
    # Cold storage next.
    if age_days > 365 and access_days > 365:
        return ARCHIVE
    if ext_cat == 1 and access_days > 180:
        return ARCHIVE  # old archives
    if size_mb > 500 and access_days > 180:
        return ARCHIVE  # big cold media
    # Warm files stay.
    if access_days <= 30:
        return KEEP
    if ext_cat == 3 and age_days <= 90:
        return KEEP
    return KEEP if access_days <= 90 else ARCHIVE


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    exts = list(EXT_MAP)
    for _ in range(N_ROWS):
        ext = rng.choice(exts)
        ext_cat = EXT_MAP[ext]
        base = rng.choice(NAME_POOLS[ext_cat])
        name = base.replace(ext, ext)  # pool already carries the extension
        # Name mutations: duplicates, version bumps, "final" stacking.
        r = rng.random()
        if r < 0.18:
            name = name.replace(ext, f"(1){ext}")
        elif r < 0.30:
            name = name.replace(ext, f"-final{ext}")
        elif r < 0.36:
            name = name.replace(ext, f"-v{rng.integers(2, 9)}{ext}")
        size_mb = float(np.exp(rng.normal(3.2, 2.4)))  # log-normal ~25 MB median
        age_days = int(rng.exponential(300))
        access_days = int(rng.exponential(200))
        hidden = int(rng.random() < 0.04)
        if hidden:
            name = f".tmp-{name}"
        clutter = clutter_score(name)
        y = label_row(size_mb, age_days, access_days, ext_cat, clutter, hidden)
        if rng.random() < NOISE_RATE:  # label noise — real life is messy
            y = int(rng.integers(0, 3))
        rows.append([round(float(np.log1p(size_mb)), 4), age_days, access_days,
                     ext_cat, round(clutter, 3), hidden, y])
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    ys = [r[-1] for r in rows]
    print(f"wrote {OUT}: {len(rows)} rows "
          f"(KEEP={ys.count(0)}, ARCHIVE={ys.count(1)}, DELETE={ys.count(2)})")


if __name__ == "__main__":
    main()
