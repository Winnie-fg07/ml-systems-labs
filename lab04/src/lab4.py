import os, random, hashlib
import numpy as np
import pandas as pd
import joblib
import yaml
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from clipper import PercentileClipper

SEED = 42
TEST_SIZE = 0.3
CLIP = (1, 99)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "..", "data")

INT_COLS = ["age", "fnlwgt", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"]
CAT_FEATURES = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
CAT_COLS = CAT_FEATURES + ["income"]
RANGES = {"age": (17, 90), "hours-per-week": (1, 99),
          "capital-gain": (0, 99999), "capital-loss": (0, 4356)}
DERIVED = ["capital_net", "hours_per_age"]

LOG = []


def log(msg):
    print(msg)
    LOG.append(str(msg))


# ---------- step 2: ingestion (raw data is never modified) ----------
def ingest():
    raw = fetch_openml(name="adult", version=2, as_frame=True, data_home=DATA).frame
    log(f"[ingest] rows={raw.shape[0]} (expected 48842), columns={raw.shape[1]} (expected 15)")
    return raw


# ---------- step 3: validation (reports problems, drops nothing) ----------
def clean_label(s):
    return s.astype(str).str.strip().str.rstrip(".")


def validate(df):
    rows = []
    for c in INT_COLS:
        s = df[c]
        ok = pd.api.types.is_integer_dtype(s) or (
            pd.api.types.is_float_dtype(s) and (s.dropna() % 1 == 0).all())
        rows.append(("type", c, 0 if ok else len(s), f"dtype={s.dtype}, expected int"))
    for c in CAT_COLS:
        ok = isinstance(df[c].dtype, pd.CategoricalDtype)
        rows.append(("type", c, 0 if ok else len(df), f"dtype={df[c].dtype}, expected category"))
    for c, (lo, hi) in RANGES.items():
        bad = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append(("range", c, bad,
                     f"allowed {lo}-{hi}, observed {df[c].min()}-{df[c].max()}"))
    for c in df.columns:
        s = df[c]
        miss = int((s.isna() | (s.astype(str).str.strip() == "?")).sum())
        rows.append(("missing", c, miss, "'?' or NaN counted as missing (allowed by policy)"))
    rows.append(("duplicates", "all columns", int(df.duplicated().sum()),
                 "exact duplicate rows, kept in raw data"))
    bad_labels = int((~clean_label(df["income"]).isin(["<=50K", ">50K"])).sum())
    rows.append(("label", "income", bad_labels, "values other than <=50K and >50K"))
    rep = pd.DataFrame(rows, columns=["check", "column", "violations", "detail"])
    rep["status"] = np.where(rep["check"].isin(["missing", "duplicates"]), "INFO",
                             np.where(rep["violations"] == 0, "PASS", "FAIL"))
    rep.to_csv(os.path.join(RESULTS, "validation_report.csv"), index=False)
    log("[validate]\n" + rep.to_string(index=False))
    return rep


# ---------- deterministic steps (no fitting, so no leakage) ----------
def mark_missing(X):
    X = X.copy()
    for c in CAT_FEATURES:
        s = X[c].astype("object")
        X[c] = s.mask(s.astype(str).str.strip() == "?", np.nan)
    return X


def add_features(X):
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
    return X


# ---------- fitted preprocessor (fitted on train only) ----------
def build_preprocessor(num_cols):
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", PercentileClipper(*CLIP)),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, CAT_FEATURES)])


def fit_and_score(Xtr, ytr, Xte, yte, num_cols):
    pre = build_preprocessor(num_cols)
    A = pre.fit_transform(Xtr)
    B = pre.transform(Xte)
    clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(A, ytr)
    pred = clf.predict(B)
    return pre, A, B, accuracy_score(yte, pred), f1_score(yte, pred, pos_label=1)


# ---------- step 6: label-quality audit ----------
def label_audit(df, y):
    feats = df.drop(columns="income").astype(str)
    feats["y"] = y.to_numpy()
    g = feats.groupby(list(feats.columns[:-1]))["y"].agg(["nunique", "size"])
    dup_groups = int((g["size"] > 1).sum())
    conflict_groups = int((g["nunique"] > 1).sum())
    conflict_rows = int(g.loc[g["nunique"] > 1, "size"].sum())
    log(f"[audit] groups of identical feature rows: {dup_groups}")
    log(f"[audit] duplicate-conflict groups (same features, different income): "
        f"{conflict_groups}, covering {conflict_rows} rows")

    edu_map = df.groupby("education", observed=True)["education-num"].agg(lambda s: s.mode()[0])
    edu_pairs = df.groupby("education", observed=True)["education-num"].nunique()
    log(f"[audit] education values mapped to more than one education-num: "
        f"{int((edu_pairs > 1).sum())}")

    sample = df.sample(50, random_state=SEED).copy()
    flags = pd.DataFrame(index=sample.index)
    for c, (lo, hi) in RANGES.items():
        flags[f"bad_{c}"] = (sample[c] < lo) | (sample[c] > hi)
    flags["edu_mismatch"] = sample["education-num"] != sample["education"].map(edu_map)
    flags["husband_female_or_wife_male"] = (
        ((sample["relationship"] == "Husband") & (sample["sex"] == "Female")) |
        ((sample["relationship"] == "Wife") & (sample["sex"] == "Male")))
    flags["gain_and_loss_both_positive"] = (
        (sample["capital-gain"] > 0) & (sample["capital-loss"] > 0))
    out = pd.concat([sample, flags], axis=1)
    out.to_csv(os.path.join(RESULTS, "label_audit_sample.csv"), index=True)
    log(f"[audit] 50-row sample flags (automatic checks): "
        f"{flags.sum().to_dict()}")
    pd.DataFrame([{"identical_feature_groups": dup_groups,
                   "conflict_groups": conflict_groups,
                   "conflict_rows": conflict_rows}]).to_csv(
        os.path.join(RESULTS, "label_audit_summary.csv"), index=False)


def dataset_stats(df, y):
    log(f"[stats] share of income >50K: {y.mean():.4f}")
    for c in ["sex", "race"]:
        log(f"[stats] {c}:\n{df[c].value_counts(normalize=True).round(4).to_string()}")
    us = (df["native-country"] == "United-States").mean()
    log(f"[stats] native-country == United-States: {us:.4f}")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    raw = ingest()
    df = raw.rename(columns={"class": "income"})

    validate(df)
    y = (clean_label(df["income"]) == ">50K").astype(int)
    X = df.drop(columns="income")
    dataset_stats(df, y)
    label_audit(df, y)

    # split BEFORE any cleaning
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    log(f"[split] train={len(Xtr)}, test={len(Xte)}")

    # exact duplicates: dropped from train only, test untouched
    dup = Xtr.assign(income=ytr).duplicated()
    log(f"[clean] exact duplicate rows dropped from train: {int(dup.sum())}")
    Xtr, ytr = Xtr[~dup], ytr[~dup]
    log(f"[clean] train rows after dedup={len(Xtr)}, test rows (untouched)={len(Xte)}")

    Xtr = add_features(mark_missing(Xtr))
    Xte = add_features(mark_missing(Xte))

    base_cols = INT_COLS
    full_cols = INT_COLS + DERIVED
    _, _, _, acc0, f10 = fit_and_score(Xtr, ytr, Xte, yte, base_cols)
    pre, A, B, acc1, f11 = fit_and_score(Xtr, ytr, Xte, yte, full_cols)
    acc0, f10, acc1, f11 = float(acc0), float(f10), float(acc1), float(f11)
    comp = pd.DataFrame([
        {"features": "original only", "accuracy": round(acc0, 4), "f1": round(f10, 4)},
        {"features": "original + capital_net + hours_per_age",
         "accuracy": round(acc1, 4), "f1": round(f11, 4)},
    ])
    comp.to_csv(os.path.join(RESULTS, "feature_comparison.csv"), index=False)
    log("[features]\n" + comp.to_string(index=False))

    joblib.dump(pre, os.path.join(RESULTS, "preprocessor.joblib"))

    # hash of the processed dataset (train, test, labels)
    h = hashlib.sha256()
    for arr in (A, B, ytr.to_numpy(), yte.to_numpy()):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    h.update(",".join(pre.get_feature_names_out()).encode())
    digest = h.hexdigest()
    log(f"[hash] SHA256 of processed dataset: {digest}")
    log(f"[hash] processed shapes: train {A.shape}, test {B.shape}")

    config = {
        "dataset": {"source": "OpenML adult, version 2 (UCI Adult / Census Income)",
                    "rows": int(raw.shape[0]), "columns": int(raw.shape[1])},
        "seed": SEED,
        "split": {"test_size": TEST_SIZE, "stratified": True},
        "cleaning": {"impute_numeric": "median", "impute_categorical": "most_frequent",
                     "clip_percentiles": list(CLIP), "scale": "StandardScaler",
                     "one_hot": "handle_unknown=ignore",
                     "duplicates_dropped_from_train": int(dup.sum())},
        "derived_features": DERIVED,
        "model": {"type": "LogisticRegression", "max_iter": 1000},
        "metrics": {"accuracy_without": round(acc0, 4), "f1_without": round(f10, 4),
                    "accuracy_with": round(acc1, 4), "f1_with": round(f11, 4)},
        "processed_shapes": {"train": list(map(int, A.shape)), "test": list(map(int, B.shape))},
        "processed_sha256": digest,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__},
    }
    with open(os.path.join(RESULTS, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    with open(os.path.join(RESULTS, "pipeline_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))


if __name__ == "__main__":
    main()