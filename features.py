"""SlothSort features: file metadata -> fixed feature vector.

SlothSort never reads file contents — only metadata the OS already exposes
(name, size, mtime/atime, extension). Everything the classifier sees is
derivable from an `ls`-style listing, which is the whole privacy story:
the file list never needs to leave the machine.
"""
import os
import re

import numpy as np

# Same category encoding as data/build_dataset.py (kept in sync deliberately).
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

FEATURES = [
    "size_mb_log",      # log1p(size in MB) — sizes span KB to GB
    "age_days",         # days since last modified
    "days_since_access",# days since last opened
    "ext_category",     # 0 installer, 1 archive, 2 media, 3 document, 4 other
    "name_clutter",     # 0..1 filename clutter signals (final/copy/v2/...)
    "is_hidden_temp",   # 0/1 dotfiles and temp names
]

CLASSES = ["KEEP", "ARCHIVE", "DELETE"]


def ext_category(filename):
    return EXT_MAP.get(os.path.splitext(filename)[1].lower(), 4)


def name_clutter(filename):
    name = filename.lower()
    hits = sum(1 for p in CLUTTER_PATTERNS if re.search(p, name))
    return min(1.0, hits / 2.0)


def is_hidden_temp(filename):
    base = os.path.basename(filename)
    return int(base.startswith(".") or base.startswith("~")
               or "temp" in base.lower() or base.lower().endswith(".tmp"))


def featurize(meta):
    """meta: dict with name, size_mb, age_days, days_since_access.
    Returns (1, 6) float64 vector in FEATURES order."""
    name = str(meta.get("name", "file.bin"))
    size_mb = max(float(meta.get("size_mb", 0.0)), 0.0)
    vec = [
        float(np.log1p(size_mb)),
        float(meta.get("age_days", 0)),
        float(meta.get("days_since_access", 0)),
        float(ext_category(name)),
        float(name_clutter(name)),
        float(is_hidden_temp(name)),
    ]
    return np.array([vec], dtype=np.float64)
