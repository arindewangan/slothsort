"""SlothSort demo dataset: a realistic SIMULATED ~/Downloads folder.

Every row is invented — names, sizes, and dates are plausible but fictional.
No real user's files were scanned. The GitHub Pages demo bakes REAL TabPFN
predictions for exactly these rows (precomputed at build time, labeled with
the build date); the Flask app scores the same rows live on request.
"""
# name, size_mb, age_days (since modified), days_since_access
DEMO_FILES = [
    {"name": "ChromeSetup.exe", "size_mb": 1.4, "age_days": 400, "days_since_access": 400},
    {"name": "node-v22.9.0.pkg", "size_mb": 32.0, "age_days": 210, "days_since_access": 210},
    {"name": "vlc-3.0.21.dmg", "size_mb": 58.0, "age_days": 340, "days_since_access": 340},
    {"name": "report-final.pdf", "size_mb": 2.1, "age_days": 12, "days_since_access": 3},
    {"name": "report-final(1).pdf", "size_mb": 2.1, "age_days": 40, "days_since_access": 120},
    {"name": "budget-v3.xlsx", "size_mb": 0.8, "age_days": 25, "days_since_access": 25},
    {"name": "Screenshot 2025-03-14 at 10.22.41.png", "size_mb": 1.2, "age_days": 220, "days_since_access": 220},
    {"name": "Screenshot 2026-10-18 at 09.01.12.png", "size_mb": 0.9, "age_days": 2, "days_since_access": 2},
    {"name": "trip-goa-2024.mp4", "size_mb": 1850.0, "age_days": 500, "days_since_access": 480},
    {"name": "IMG_2041.HEIC", "size_mb": 3.4, "age_days": 90, "days_since_access": 20},
    {"name": "project-backup-2023.zip", "size_mb": 640.0, "age_days": 700, "days_since_access": 700},
    {"name": "old-invoices.tar.gz", "size_mb": 120.0, "age_days": 800, "days_since_access": 800},
    {"name": "song-demo-final.mp3", "size_mb": 8.0, "age_days": 150, "days_since_access": 150},
    {"name": "presentation-final-FINAL.pptx", "size_mb": 14.0, "age_days": 60, "days_since_access": 200},
    {"name": "notes.txt", "size_mb": 0.02, "age_days": 5, "days_since_access": 1},
    {"name": ".DS_Store", "size_mb": 0.006, "age_days": 300, "days_since_access": 300},
    {"name": "~$temp-doc.docx", "size_mb": 0.05, "age_days": 95, "days_since_access": 95},
    {"name": "dataset-kaggle-v2.csv", "size_mb": 320.0, "age_days": 260, "days_since_access": 260},
    {"name": "ebook-python.epub", "size_mb": 12.0, "age_days": 180, "days_since_access": 45},
    {"name": "wallpaper-4k.jpg", "size_mb": 9.0, "age_days": 400, "days_since_access": 400},
    {"name": "resume-2026.pdf", "size_mb": 0.4, "age_days": 30, "days_since_access": 10},
    {"name": "tax-docs-2024.zip", "size_mb": 45.0, "age_days": 380, "days_since_access": 380},
    {"name": "movie-night.mkv", "size_mb": 4200.0, "age_days": 100, "days_since_access": 95},
    {"name": "Untitled-1.txt", "size_mb": 0.001, "age_days": 130, "days_since_access": 130},
]

SIMULATED_NOTE = ("SIMULATED dataset — every filename, size, and date above is "
                  "invented. No real files were scanned.")
