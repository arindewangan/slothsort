"""SlothSort recommender: predicted class -> action + human-readable reasons.

The predicted class comes from the classifier (TabPFN or the labeled
fallback). The reasons are deterministic, metadata-derived explanations —
they describe WHAT the file looks like (untouched for N days, installer,
duplicate-y name), never invented model internals. This keeps the UI honest:
the model decides the class, the file's own metadata explains it.
"""
from features import ext_category, name_clutter, CLASSES

EXT_NAMES = {0: "installer", 1: "archive", 2: "media file",
             3: "document", 4: "file"}

ACTIONS = {
    "KEEP": ("Keep it", "Nothing to do — this file earns its disk space."),
    "ARCHIVE": ("Archive it", "Move to cold storage: out of Downloads, still retrievable."),
    "DELETE": ("Delete it", "Safe to remove — re-downloadable or pure clutter."),
}


def _fmt_size(size_mb):
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    if size_mb >= 1:
        return f"{size_mb:.0f} MB"
    return f"{size_mb * 1024:.0f} KB"


def reasons_for(meta, prediction):
    """List of short human-readable reasons, derived from the file's metadata."""
    name = str(meta.get("name", "file"))
    size_mb = float(meta.get("size_mb", 0.0))
    age = int(meta.get("age_days", 0))
    access = int(meta.get("days_since_access", 0))
    cat = ext_category(name)
    clutter = name_clutter(name)
    reasons = []
    kind = EXT_NAMES[cat]
    if access >= 365:
        reasons.append(f"Untouched for {access} days — over a year of neglect")
    elif access >= 180:
        reasons.append(f"Not opened in {access} days")
    elif access <= 30:
        reasons.append(f"Opened {access} days ago — still in active use")
    if prediction == "DELETE":
        if cat == 0:
            reasons.append("It's an installer — you can re-download it any time")
        if clutter >= 0.5:
            reasons.append(f"Suspicious name ({name!r}) — smells like a duplicate")
        if name.lower().startswith((".tmp", "~")) or "temp" in name.lower():
            reasons.append("Looks like a temp/scratch file")
    if prediction == "ARCHIVE":
        reasons.append(f"Cold {kind} ({_fmt_size(size_mb)}) — worth keeping, not worth the clutter")
        if age >= 365:
            reasons.append(f"Sitting here for {age} days")
    if prediction == "KEEP" and cat == 3 and age <= 90:
        reasons.append("Recent document — probably still relevant")
    if size_mb >= 500 and prediction in ("ARCHIVE", "DELETE"):
        reasons.append(f"Frees {_fmt_size(size_mb)} of disk")
    if not reasons:
        reasons.append(f"{kind.capitalize()} with no strong clutter signals")
    return reasons[:3]


def recommend(meta, classification):
    """Combine classifier output with action + reasons. JSON-safe."""
    pred = classification["prediction"]
    action, detail = ACTIONS[pred]
    return {
        "prediction": pred,
        "action": action,
        "action_detail": detail,
        "reasons": reasons_for(meta, pred),
        "confidence": classification["confidence"],
        "confidence_pct": classification["confidence_pct"],
        "probabilities": classification["probabilities"],
        "needs_review": classification["needs_review"],
        "model": classification["model"],
        "model_kind": classification["model_kind"],
        "data_note": classification["data_note"],
    }
