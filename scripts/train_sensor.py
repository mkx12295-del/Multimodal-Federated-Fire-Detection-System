"""Train the XGBoost sensor model with a leakage-free chronological split.

Also reports what the v1 random row split would have scored, to show how much
that split overstated accuracy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from firenode.features import FEATURES, chronological_split, load_iot_csv, xy  # noqa: E402


def scores(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


def make_model(**kw):
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                  colsample_bytree=1.0, random_state=42, eval_metric="logloss")
    params.update(kw)
    return xgb.XGBClassifier(**params)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=str(ROOT / "smoke_detection_iot.csv"))
    p.add_argument("--out", default=str(ROOT / "models" / "sensor_xgb.json"))
    args = p.parse_args()

    df = load_iot_csv(args.csv)

    # v1 protocol: random 80/20 row split (adjacent seconds end up on both sides).
    X, y = xy(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    v1 = make_model(n_estimators=50, learning_rate=0.1, subsample=1.0).fit(Xtr, ytr)
    random_split = scores(yte, v1.predict_proba(Xte)[:, 1])

    # v2 protocol: chronological split inside every recording session.
    parts = chronological_split(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = (xy(parts[k]) for k in ("train", "val", "test"))
    model = make_model(early_stopping_rounds=30)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    model.get_booster().feature_names = FEATURES
    chrono = scores(yte, model.predict_proba(Xte)[:, 1])

    # Leave-one-session-out: train on every other recording session, test on the
    # held-out one. This approximates deploying the model at a new site.
    loso = {}
    for s in sorted(df["session"].unique()):
        tr, te = df[df["session"] != s], df[df["session"] == s]
        m = make_model(n_estimators=model.best_iteration + 1).fit(*xy(tr))
        Xs, ys = xy(te)
        pred = m.predict(Xs)
        loso[f"session_{s}"] = {"rows": int(len(te)), "fire_rate": float(ys.mean()),
                                "accuracy": accuracy_score(ys, pred)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))

    report = {
        "features": FEATURES,
        "rows": {k: int(len(v)) for k, v in parts.items()},
        "best_iteration": int(model.best_iteration),
        "random_row_split_v1": random_split,
        "chronological_split_v2": chrono,
        "leave_one_session_out": loso,
        "feature_importance": dict(zip(FEATURES, map(float, model.feature_importances_))),
    }
    res = ROOT / "results" / "sensor_metrics.json"
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved model to {out}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
