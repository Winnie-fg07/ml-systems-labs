import os, random, time, statistics
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
INIT_STD = 0.1   # N(0, 0.01) read as variance 0.01, so std = 0.1
BATCH = 32
EPOCHS = 50
LRS = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


# ---------- data ----------
def load_data():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Y = np.eye(10)[y]
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=42
    )
    return X[idx_train], X[idx_test], Y[idx_train], Y[idx_test]


# ---------- network ----------
def init_params(width, rng):
    sizes = [(64, width), (width, width), (width, 10)]
    p = {}
    for i, (a, b) in enumerate(sizes, start=1):
        p[f"W{i}"] = rng.normal(0.0, INIT_STD, size=(a, b))
        p[f"b{i}"] = np.zeros(b)
    return p


def forward(p, X):
    Z1 = X @ p["W1"] + p["b1"]
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = np.maximum(0, Z2)
    Z3 = A2 @ p["W3"] + p["b3"]
    Z3 = Z3 - Z3.max(axis=1, keepdims=True)   # for numerical stability
    E = np.exp(Z3)
    P = E / E.sum(axis=1, keepdims=True)
    return P, (X, Z1, A1, Z2, A2)


def cross_entropy(P, Y):
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1.0)), axis=1)))


def backward(p, cache, P, Y):
    X, Z1, A1, Z2, A2 = cache
    B = X.shape[0]
    g = {}
    dZ3 = (P - Y) / B
    g["W3"] = A2.T @ dZ3
    g["b3"] = dZ3.sum(axis=0)
    dA2 = dZ3 @ p["W3"].T
    dZ2 = dA2 * (Z2 > 0)
    g["W2"] = A1.T @ dZ2
    g["b2"] = dZ2.sum(axis=0)
    dA1 = dZ2 @ p["W2"].T
    dZ1 = dA1 * (Z1 > 0)
    g["W1"] = X.T @ dZ1
    g["b1"] = dZ1.sum(axis=0)
    return g


def accuracy(P, Y):
    return float(np.mean(np.argmax(P, axis=1) == np.argmax(Y, axis=1)))


# ---------- checks ----------
def sanity_checks(Xtr, Ytr):
    rows = []
    for w in WIDTHS:
        p = init_params(w, np.random.default_rng(SEED))
        shapes = {k: v.shape for k, v in p.items()}
        print(f"Width {w} parameter shapes: {shapes}")
        P, _ = forward(p, Xtr)
        rows.append({
            "width": w,
            "output_shape": str(P.shape),
            "max_row_sum_error": float(np.abs(P.sum(axis=1) - 1).max()),
            "initial_loss": round(cross_entropy(P, Ytr), 4),
            "ln10": round(float(np.log(10)), 4),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(RESULTS, "sanity_checks.csv"), index=False)


def gradient_check(width, X, Y, n_checks=10, eps=1e-5):
    rng = np.random.default_rng(SEED)
    p = init_params(width, rng)
    P, cache = forward(p, X)
    g = backward(p, cache, P, Y)

    def loss_now():
        Pn, _ = forward(p, X)
        return cross_entropy(Pn, Y)

    names = ["W1", "W2", "W3"]
    rows = []
    while len(rows) < n_checks:
        name = names[rng.integers(3)]
        i = int(rng.integers(p[name].shape[0]))
        j = int(rng.integers(p[name].shape[1]))
        old = p[name][i, j]
        p[name][i, j] = old + eps
        lp = loss_now()
        p[name][i, j] = old - eps
        lm = loss_now()
        p[name][i, j] = old
        numeric = (lp - lm) / (2 * eps)
        analytic = float(g[name][i, j])
        if analytic == 0.0:
            continue
        rel = abs(analytic - numeric) / max(abs(analytic), abs(numeric), 1e-12)
        rows.append({
            "width": width, "weight": f"{name}[{i},{j}]",
            "analytic": analytic, "numeric": numeric,
            "rel_error": rel, "pass": rel < 1e-6,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"Width {width}: {int(df['pass'].sum())} of {n_checks} passed (need at least 9)")
    return df


# ---------- training ----------
def train(width, lr, Xtr, Ytr, Xte, Yte):
    rng = np.random.default_rng(SEED)
    p = init_params(width, rng)
    n = Xtr.shape[0]
    log = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)
        for s in range(0, n, BATCH):
            b = order[s:s + BATCH]
            P, cache = forward(p, Xtr[b])
            g = backward(p, cache, P, Ytr[b])
            for k in p:
                p[k] -= lr * g[k]
        Ptr, _ = forward(p, Xtr)
        Pte, _ = forward(p, Xte)
        rec = {
            "width": width, "lr": lr, "epoch": epoch,
            "train_loss": cross_entropy(Ptr, Ytr),
            "test_loss": cross_entropy(Pte, Yte),
            "test_acc": accuracy(Pte, Yte),
        }
        log.append(rec)
        print(f"W={width} lr={lr} epoch {epoch:2d}/{EPOCHS} "
              f"train_loss={rec['train_loss']:.4f} test_acc={rec['test_acc']:.4f}")
    return log


# ---------- timing ----------
def time_passes(width, X, Y, runs=100):
    p = init_params(width, np.random.default_rng(SEED))
    P, cache = forward(p, X)          # warm-up
    backward(p, cache, P, Y)
    fwd, bwd = [], []
    for _ in range(runs):
        t = time.perf_counter()
        P, cache = forward(p, X)
        fwd.append((time.perf_counter() - t) * 1e6)
        t = time.perf_counter()
        backward(p, cache, P, Y)
        bwd.append((time.perf_counter() - t) * 1e6)
    return statistics.median(fwd), statistics.median(bwd)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS, exist_ok=True)

    Xtr, Xte, Ytr, Yte = load_data()
    print("Train:", Xtr.shape, Ytr.shape, "Test:", Xte.shape, Yte.shape)

    print("\n=== Forward pass and loss sanity checks ===")
    sanity_checks(Xtr, Ytr)

    print("\n=== Gradient checking ===")
    gc = pd.concat([gradient_check(w, Xtr[:32], Ytr[:32]) for w in WIDTHS])
    gc.to_csv(os.path.join(RESULTS, "gradient_check.csv"), index=False)

    print("\n=== Training ===")
    all_logs = []
    for w in WIDTHS:
        for lr in LRS:
            all_logs += train(w, lr, Xtr, Ytr, Xte, Yte)
    logs = pd.DataFrame(all_logs)
    logs.to_csv(os.path.join(RESULTS, "training_log.csv"), index=False)

    final = logs[logs["epoch"] == EPOCHS][
        ["width", "lr", "train_loss", "test_loss", "test_acc"]
    ].round(4)
    final.to_csv(os.path.join(RESULTS, "final_accuracy.csv"), index=False)
    print("\nFinal results:")
    print(final.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, w in zip(axes, WIDTHS):
        for lr in LRS:
            d = logs[(logs["width"] == w) & (logs["lr"] == lr)]
            ax.plot(d["epoch"], d["train_loss"], label=f"lr={lr}")
        ax.set_title(f"Hidden width {w}")
        ax.set_xlabel("Epoch")
        ax.legend()
    axes[0].set_ylabel("Training loss (cross-entropy)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "loss_curves.png"), dpi=150)

    print("\n=== Timing (batch of 32, median of 100 runs, microseconds) ===")
    rows = []
    for w in WIDTHS:
        f, b = time_passes(w, Xtr[:32], Ytr[:32])
        rows.append({"width": w, "forward_us": round(f, 2),
                     "backward_us": round(b, 2), "ratio_bwd_over_fwd": round(b / f, 2)})
    tdf = pd.DataFrame(rows)
    tdf.to_csv(os.path.join(RESULTS, "timing.csv"), index=False)
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()