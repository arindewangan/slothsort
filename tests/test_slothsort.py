"""SlothSort tests — real classifier inference on the simulated dataset; no invented results.

TabPFN-dependent tests are skipped (with a clear message) when TabPFN is not
installed; the fallback-mode tests still run and assert the fallback is
honestly labeled.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import FEATURES, CLASSES, featurize, ext_category, name_clutter
from classifier import (get_model, load_dataset, classify, classify_batch,
                        tabpfn_available, model_mode)
from recommender import recommend, reasons_for
from demo_data import DEMO_FILES

tabpfn_only = pytest.mark.skipif(
    not tabpfn_available() or model_mode() != "tabpfn",
    reason="TabPFN not installed/selected — honest fallback mode")


def file_meta(**kw):
    base = {"name": "setup-old.exe", "size_mb": 85.0, "age_days": 300,
            "days_since_access": 300}
    base.update(kw)
    return base


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def model():
    return get_model()


def test_dataset_loads_with_expected_shape(dataset):
    X, y = dataset
    assert X.shape[1] == len(FEATURES)
    assert len(X) == len(y) >= 5000
    assert set(np.unique(y)) <= {0, 1, 2}


def test_dataset_has_all_three_classes(dataset):
    _, y = dataset
    counts = np.bincount(y)
    assert len(counts) == 3
    assert (counts > 200).all(), f"class imbalance too extreme: {counts}"


def test_model_caches(model):
    assert get_model() is model


def test_model_kind_is_honest(model):
    if tabpfn_available() and model_mode() == "tabpfn":
        assert model["kind"] == "tabpfn"
    else:
        assert model["kind"] == "fallback"


def test_featurize_order_and_defaults():
    X = featurize({"name": "x.pdf"})
    assert X.shape == (1, len(FEATURES))
    assert np.isfinite(X).all()


def test_ext_category_known_and_unknown():
    assert ext_category("setup.exe") == 0
    assert ext_category("data.zip") == 1
    assert ext_category("clip.mp4") == 2
    assert ext_category("doc.pdf") == 3
    assert ext_category("weird.xyz") == 4


def test_name_clutter_signals():
    assert name_clutter("report-final-FINAL(1).pdf") > name_clutter("report.pdf")
    assert 0.0 <= name_clutter("anything") <= 1.0


def test_classify_schema():
    r = classify(file_meta())
    for k in ("prediction", "confidence", "confidence_pct", "probabilities",
              "model", "model_kind", "needs_review", "context_rows", "data_note"):
        assert k in r, k
    assert r["prediction"] in CLASSES
    assert 0.0 <= r["confidence"] <= 1.0
    assert abs(sum(r["probabilities"].values()) - 1.0) < 0.01
    assert "SIMULATED" in r["data_note"]
    if r["model_kind"] == "fallback":
        assert "NOT TabPFN" in r["model"]
    else:
        assert "TabPFN" in r["model"]


def test_recommend_schema_and_reasons():
    r = recommend(file_meta(), classify(file_meta()))
    assert r["prediction"] in CLASSES
    assert r["action"] and r["action_detail"]
    assert 1 <= len(r["reasons"]) <= 3
    assert all(isinstance(x, str) and x for x in r["reasons"])


def test_reasons_mention_real_metadata():
    # Reasons must reference the file's own observable facts, not invented internals.
    meta = file_meta(name="ChromeSetup.exe", age_days=400, days_since_access=400)
    rs = reasons_for(meta, "DELETE")
    blob = " ".join(rs)
    assert "400" in blob or "installer" in blob.lower()


def test_json_serializable():
    json.dumps(recommend(file_meta(), classify(file_meta())))


@tabpfn_only
def test_tabpfn_separates_obvious_cases():
    junk = file_meta(name="setup-old-v2.exe", size_mb=90, age_days=400,
                     days_since_access=400)
    fresh = file_meta(name="report.pdf", size_mb=2, age_days=5, days_since_access=1)
    pj, pf = classify(junk)["probabilities"], classify(fresh)["probabilities"]
    # Directional sanity on simulated data (not a metric claim).
    assert pj["DELETE"] > pf["DELETE"]
    assert pf["KEEP"] > pj["KEEP"]


@tabpfn_only
def test_tabpfn_context_size():
    # 333 per class x 3 classes = 999-row stratified context (see classifier.py)
    assert get_model()["n_train"] == 999


def test_classify_batch_matches_single():
    metas = [file_meta(name=f"f{i}.pdf", age_days=10 * i, days_since_access=5 * i)
             for i in range(4)]
    batched = classify_batch(metas)
    assert len(batched) == 4
    for meta, b in zip(metas, batched):
        s = classify(meta)
        assert b["prediction"] == s["prediction"]
        assert b["probabilities"] == s["probabilities"]


def test_demo_files_all_classifiable():
    for meta in DEMO_FILES:
        r = recommend(meta, classify(meta))
        assert r["prediction"] in CLASSES
        assert r["confidence_pct"] >= 0


def test_flask_health_ok():
    from app import app as flask_app
    c = flask_app.test_client()
    r = c.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["demo_files"] == len(DEMO_FILES)


def test_flask_index_ok():
    from app import app as flask_app
    c = flask_app.test_client()
    r = c.get("/")
    assert r.status_code == 200
    assert b"SlothSort" in r.data


def test_api_scan_ok():
    from app import app as flask_app
    c = flask_app.test_client()
    r = c.post("/api/scan", json={"files": [file_meta()]})
    assert r.status_code == 200
    body = r.get_json()
    assert body["simulated_input"] is True
    assert body["results"][0]["prediction"] in CLASSES


def test_api_scan_rejects_empty():
    from app import app as flask_app
    c = flask_app.test_client()
    r = c.post("/api/scan", json={"files": []})
    assert r.status_code == 400


def test_api_demo_marks_simulated():
    from app import app as flask_app
    c = flask_app.test_client()
    r = c.get("/api/demo")
    body = r.get_json()
    assert body["simulated"] is True
    assert len(body["results"]) == len(DEMO_FILES)


def test_api_act_archive_delete_undo():
    from app import app as flask_app
    c = flask_app.test_client()
    name = DEMO_FILES[0]["name"]
    r = c.post("/api/act", json={"action": "archive", "name": name})
    assert r.get_json()["new_state"] == "archived"
    r = c.post("/api/act", json={"action": "delete", "name": name})
    assert r.get_json()["new_state"] == "deleted"
    r = c.post("/api/act", json={"action": "undo"})
    assert r.get_json()["ok"] is True
    assert r.get_json()["state"][name] == "archived"
    r = c.post("/api/act", json={"action": "undo"})
    assert r.get_json()["state"][name] == "active"


def test_api_act_rejects_bad_action():
    from app import app as flask_app
    c = flask_app.test_client()
    r = c.post("/api/act", json={"action": "format", "name": "x"})
    assert r.status_code == 400
