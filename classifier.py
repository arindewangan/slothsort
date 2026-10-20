"""SlothSort classifier: TabPFN (open-source tabular foundation model) on CPU.

Mode is chosen by the TABPFN_MODEL env var:
  - "tabpfn" (default): real TabPFNClassifier inference, local CPU, no keys.
  - "fallback": a labeled rule-based fallback (used only when TabPFN is not
    installed / weights can't load). Anything produced in fallback mode is
    explicitly labeled as such — never presented as a TabPFN prediction.

TabPFN does in-context learning: "fit" just stores a seeded, class-stratified
999-row context (333 per class) of the simulated dataset (attention is O(n^2)
in context size, so ~1000 rows keeps per-request CPU inference in the
tens-of-seconds range on this box).
n_estimators=2 keeps the ensemble honest while staying CPU-friendly.
"""
import csv
import os

import numpy as np

from features import FEATURES, featurize, CLASSES

DATA_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "file_history.csv")
MODEL_ENV = "TABPFN_MODEL"
CONTEXT_ROWS = 1000
SEED = 7

_cache = {}


def model_mode():
    return os.environ.get(MODEL_ENV, "tabpfn").lower()


def tabpfn_available():
    try:
        import tabpfn  # noqa: F401
        return True
    except Exception:
        return False


def load_dataset():
    with open(DATA_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    return X, y


def _fit_tabpfn():
    from tabpfn import TabPFNClassifier

    X, y = load_dataset()
    rng = np.random.default_rng(SEED)
    keep = []
    for cls in (0, 1, 2):
        idx = np.flatnonzero(y == cls)
        n = min(CONTEXT_ROWS // 3, len(idx))
        keep.append(rng.choice(idx, size=n, replace=False))
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    Xc, yc = X[keep], y[keep]
    clf = TabPFNClassifier(device="cpu", n_estimators=2)
    clf.fit(Xc, yc)
    return {"kind": "tabpfn", "clf": clf, "n_train": len(Xc),
            "class_counts": {CLASSES[c]: int((yc == c).sum()) for c in (0, 1, 2)}}


def _fit_fallback():
    # Honest labeled fallback: the same heuristics that labeled the training
    # data, returned as probabilities with a clear label. NOT TabPFN.
    return {"kind": "fallback", "n_train": 0, "class_counts": {}}


def get_model():
    """Lazily build the classifier; respects TABPFN_MODEL and availability."""
    if "model" not in _cache:
        mode = model_mode()
        if mode == "tabpfn" and tabpfn_available():
            _cache["model"] = _fit_tabpfn()
        else:
            _cache["model"] = _fit_fallback()
            _cache["model"]["requested"] = mode
    return _cache["model"]


def _fallback_proba(X):
    # Mirror of the dataset labeling rules, softened to probabilities.
    out = []
    for row in X:
        size_log, age, access, ext_cat, clutter, hidden = row
        size_mb = float(np.expm1(size_log))
        if hidden >= 1 or (ext_cat == 0 and access > 180) or \
           (clutter > 0.6 and age > 120 and access > 120):
            out.append([0.05, 0.10, 0.85])
        elif (age > 365 and access > 365) or (ext_cat == 1 and access > 180) or \
             (size_mb > 500 and access > 180):
            out.append([0.10, 0.80, 0.10])
        else:
            out.append([0.80, 0.12, 0.08])
    return np.array(out)


def predict_proba(X):
    m = get_model()
    if m["kind"] == "tabpfn":
        return np.asarray(m["clf"].predict_proba(X), dtype=float)
    return _fallback_proba(X)


def _pack_result(m, proba):
    order = np.argsort(proba)[::-1]
    top = int(order[0])
    conf = float(proba[top])
    return {
        "prediction": CLASSES[top],
        "confidence": round(conf, 3),
        "confidence_pct": round(conf * 100, 1),
        "probabilities": {CLASSES[i]: round(float(proba[i]), 3) for i in range(3)},
        "model": ("TabPFN (open-source tabular foundation model, local CPU)"
                  if m["kind"] == "tabpfn"
                  else "fallback rule-based classifier (labeled fallback — NOT TabPFN)"),
        "model_kind": m["kind"],
        "needs_review": bool(conf < 0.55),
        "context_rows": m["n_train"],
        "data_note": "SIMULATED training data — synthetic file-history records generated locally; not real user files.",
    }


def classify(meta):
    """Classify one file's metadata dict. Returns JSON-safe result dict."""
    m = get_model()
    proba = predict_proba(featurize(meta))[0]
    return _pack_result(m, proba)


def classify_batch(metas):
    """Classify many files in ONE forward pass (TabPFN scores query rows
    independently against the same context, so batch == sequential)."""
    m = get_model()
    metas = list(metas)
    X = np.vstack([featurize(mt)[0] for mt in metas])
    probas = predict_proba(X)
    return [_pack_result(m, probas[i]) for i in range(len(metas))]
