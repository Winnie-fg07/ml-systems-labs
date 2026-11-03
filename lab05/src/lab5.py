import os, sys, random, time, statistics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from engine import Value

SEED = 42
LR = 0.5
EPOCHS = 100
HIDDEN = 8
EPS = 1e-5
WARMUP = 20

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def out_path(name):
    return os.path.join(RESULTS, name)


# =====================================================================
# Task 4: engine vs PyTorch autograd vs numerical gradients
# =====================================================================
def graph_summary(root):
    seen, stack, ops = set(), [root], {}
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        key = n._op or "leaf"
        ops[key] = ops.get(key, 0) + 1
        stack.extend(n._parents)
    return len(seen), ops


def reuse_check():
    x = Value(3.0)
    y = x * x + x            # x is used 3 times, so its gradient must accumulate
    y.backward()
    ok = abs(x.grad - 7.0) < 1e-12
    print(f"Accumulation check: y = x*x + x at x=3, dy/dx should be 7, "
          f"engine gives {x.grad} -> {'PASS' if ok else 'FAIL'}")
    return ok


def verify_gradients():
    print("\n=== Task 4: gradient verification ===")
    reuse_check()
    a0, b0, c0 = [2.0, -1.5, 0.5], [-3.0, 2.0, 4.0], [1.0, 0.5, -1.0]
    n = len(a0)

    # my engine
    a, b, c = [Value(v) for v in a0], [Value(v) for v in b0], [Value(v) for v in c0]
    out = sum(((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c))
    nodes, ops = graph_summary(out)
    print(f"Graph built during the forward pass: {nodes} nodes, operations: {ops}")
    out.backward()
    eng = np.array([v.grad for v in a + b + c])

    # PyTorch autograd (float64 so the comparison is fair)
    ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True)
                  for v in (a0, b0, c0))
    tout = ((ta * tb + tc) ** 2).relu().sum()
    tout.backward()
    tor = np.concatenate([ta.grad.numpy(), tb.grad.numpy(), tc.grad.numpy()])
    print(f"Forward value: engine {out.data:.10f}, torch {tout.item():.10f}")

    # numerical gradients (central differences)
    base = np.array(a0 + b0 + c0)

    def f(v):
        return float(np.maximum((v[:n] * v[n:2 * n] + v[2 * n:]) ** 2, 0.0).sum())

    num = np.zeros_like(base)
    for i in range(len(base)):
        up, dn = base.copy(), base.copy()
        up[i] += EPS
        dn[i] -= EPS
        num[i] = (f(up) - f(dn)) / (2 * EPS)

    names = [f"{s}{i}" for s in "abc" for i in range(n)]
    df = pd.DataFrame({"input": names, "engine": eng, "torch": tor, "numerical": num})
    df.to_csv(out_path("gradient_check.csv"), index=False)
    print(df.to_string(index=False))

    d_et = float(np.abs(eng - tor).max())
    d_en = float(np.abs(eng - num).max())
    d_tn = float(np.abs(tor - num).max())
    summary = pd.DataFrame([
        {"comparison": "engine vs torch", "max_abs_diff": d_et, "pass": d_et < 1e-6},
        {"comparison": "engine vs numerical", "max_abs_diff": d_en, "pass": d_en < 1e-6},
        {"comparison": "torch vs numerical", "max_abs_diff": d_tn, "pass": d_tn < 1e-6},
    ])
    summary.to_csv(out_path("gradient_check_summary.csv"), index=False)
    print(summary.to_string(index=False))


# =====================================================================
# Task 5: tiny model on make_moons, my engine vs PyTorch
# =====================================================================
def load_moons():
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


def init_params():
    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0.0, 0.5, (2, HIDDEN))
    W2 = rng.normal(0.0, 0.5, (HIDDEN, 1))
    return W1, np.zeros(HIDDEN), W2, np.zeros(1)


def predict_acc(weights, X, y):
    W1, b1, W2, b2 = weights
    h = np.maximum(X @ W1 + b1, 0.0)
    z = (h @ W2 + b2).ravel()
    return float(((z > 0).astype(int) == y).mean())


def train_engine(X, y, epochs):
    W1, b1, W2, b2 = init_params()
    w1 = [[Value(W1[i, j]) for j in range(HIDDEN)] for i in range(2)]
    bias1 = [Value(0.0) for _ in range(HIDDEN)]
    w2 = [Value(W2[j, 0]) for j in range(HIDDEN)]
    bias2 = Value(0.0)
    params = [q for row in w1 for q in row] + bias1 + w2 + [bias2]
    xs, ys, n = X.tolist(), y.tolist(), len(y)
    losses = []
    for _ in range(epochs):
        terms = []
        for (x0, x1), t in zip(xs, ys):
            hidden = [(w1[0][j] * x0 + w1[1][j] * x1 + bias1[j]).relu()
                      for j in range(HIDDEN)]
            z = sum(h * w for h, w in zip(hidden, w2)) + bias2
            p = z.sigmoid()
            terms.append(-(t * p.log() + (1 - t) * (1 - p).log()))   # BCE
        loss = sum(terms) * (1.0 / n)
        for q in params:
            q.grad = 0.0
        loss.backward()
        for q in params:
            q.data -= LR * q.grad
        losses.append(loss.data)
    weights = (np.array([[w1[i][j].data for j in range(HIDDEN)] for i in range(2)]),
               np.array([v.data for v in bias1]),
               np.array([[v.data] for v in w2]),
               np.array([bias2.data]))
    return losses, weights


def train_torch(X, y, epochs):
    W1, b1, W2, b2 = init_params()
    ps = [torch.tensor(v, dtype=torch.float64, requires_grad=True)
          for v in (W1, b1, W2, b2)]
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    opt = torch.optim.SGD(ps, lr=LR)
    losses = []
    for _ in range(epochs):
        h = torch.relu(Xt @ ps[0] + ps[1])
        z = (h @ ps[2] + ps[3]).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(z, yt)   # sigmoid + BCE together
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, tuple(p.detach().numpy() for p in ps)


def moons_experiment():
    print("\n=== Task 5: make_moons, my engine vs PyTorch ===")
    print(f"Architecture 2 -> {HIDDEN} (ReLU) -> 1 (sigmoid), BCE loss, "
          f"full-batch SGD, lr={LR}, {EPOCHS} epochs. This part takes a few minutes.")
    Xtr, Xte, ytr, yte = load_moons()
    ytr_f = ytr.astype(float)
    train_engine(Xtr, ytr_f, 2)          # warm-up
    train_torch(Xtr, ytr_f, 2)
    et, tt = [], []
    for r in range(5):
        t = time.perf_counter()
        e_loss, e_w = train_engine(Xtr, ytr_f, EPOCHS)
        et.append(time.perf_counter() - t)
        t = time.perf_counter()
        t_loss, t_w = train_torch(Xtr, ytr_f, EPOCHS)
        tt.append(time.perf_counter() - t)
        print(f"run {r + 1}/5: engine {et[-1]:.2f} s, torch {tt[-1]:.4f} s")
    med_e, med_t = statistics.median(et), statistics.median(tt)
    res = pd.DataFrame([
        {"implementation": "my engine", "median_time_s": round(med_e, 4),
         "final_train_loss": round(e_loss[-1], 6),
         "test_acc": round(predict_acc(e_w, Xte, yte), 4)},
        {"implementation": "PyTorch", "median_time_s": round(med_t, 4),
         "final_train_loss": round(t_loss[-1], 6),
         "test_acc": round(predict_acc(t_w, Xte, yte), 4)},
    ])
    res.to_csv(out_path("moons_comparison.csv"), index=False)
    print(res.to_string(index=False))
    print(f"Engine is {med_e / med_t:.1f} times slower than PyTorch (median of 5 runs).")
    print(f"Max abs difference between the two loss curves: "
          f"{float(np.abs(np.array(e_loss) - np.array(t_loss)).max()):.3e}")
    plt.figure(figsize=(6, 4))
    plt.plot(e_loss, label="my engine")
    plt.plot(t_loss, "--", label="PyTorch")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (BCE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path("moons_loss.png"), dpi=150)


# =====================================================================
# Task 6: PyTorch graph, eager vs traced/compiled, contiguity
# =====================================================================
def median_ms(fn, runs, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def make_model():
    return nn.Sequential(
        nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)).eval()


def walk(fn, lines, depth=0, seen=None):
    seen = set() if seen is None else seen
    tag = fn.name()
    if id(fn) in seen:
        lines.append("  " * depth + tag + " (already shown)")
        return
    seen.add(id(fn))
    lines.append("  " * depth + tag)
    for nxt, _ in fn.next_functions:
        if nxt is not None:
            walk(nxt, lines, depth + 1, seen)


def pytorch_experiments():
    print("\n=== Task 6: PyTorch graph, eager vs traced/compiled, contiguity ===")
    print("torch threads:", torch.get_num_threads())
    model = make_model()
    x = torch.randn(1, 1, 28, 28)

    # --- inspect the autograd graph ---
    out = model(x).sum()
    lines = []
    walk(out.grad_fn, lines)
    with open(out_path("autograd_graph.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Autograd graph (output first):\n" + "\n".join(lines))

    # --- inspect tensors: shape, dtype, contiguity ---
    w = model[1].weight
    items = {"input x": x, "Linear weight": w, "Linear weight .t() (view)": w.t(),
             "hidden activation": model[:2](x)}
    props = pd.DataFrame([{"tensor": k, "shape": str(tuple(v.shape)),
                           "dtype": str(v.dtype), "stride": str(v.stride()),
                           "contiguous": v.is_contiguous()} for k, v in items.items()])
    props.to_csv(out_path("tensor_properties.csv"), index=False)
    print(props.to_string(index=False))

    # --- trace (always works) and try torch.compile (may be unsupported on Windows) ---
    with torch.no_grad():
        traced = torch.jit.trace(model, x).eval()
    with open(out_path("traced_graph.txt"), "w", encoding="utf-8") as f:
        f.write(str(traced.graph))

    compiled, note = None, ""
    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled(x)
    except Exception as e:
        compiled = None
        note = (f"torch.compile failed: {type(e).__name__}: "
                f"{(str(e).splitlines() or [''])[0][:150]}")
        print(note)

    rows = []
    with torch.no_grad():
        for batch, runs in ((1, 100), (128, 20)):
            xb = torch.randn(batch, 1, 28, 28)
            variants = [("eager", model), ("jit.trace", traced)]
            if compiled is not None:
                variants.append(("torch.compile", compiled))
            for name, m in variants:
                try:
                    ms = median_ms(lambda: m(xb), runs, WARMUP)
                    rows.append({"variant": name, "batch": batch, "runs": runs,
                                 "median_ms": round(ms, 4), "note": ""})
                except Exception as e:
                    rows.append({"variant": name, "batch": batch, "runs": runs,
                                 "median_ms": None, "note": type(e).__name__})
    if compiled is None:
        rows.append({"variant": "torch.compile", "batch": "-", "runs": "-",
                     "median_ms": None, "note": note})
    ev = pd.DataFrame(rows)
    eager = ev[ev["variant"] == "eager"].set_index("batch")["median_ms"]
    ev["speedup_vs_eager"] = [
        round(eager[r["batch"]] / r["median_ms"], 3)
        if r["batch"] in eager.index and r["median_ms"] else None
        for _, r in ev.iterrows()]
    ev.to_csv(out_path("eager_vs_compiled.csv"), index=False)
    print(ev.to_string(index=False))

    # --- contiguity: how a non-contiguous view changes operation speed ---
    A = torch.randn(2048, 2048)
    cases = {"transpose view A.t()": A.t(), "strided slice A[:, ::2]": A[:, ::2]}
    crow = []
    for cname, v in cases.items():
        c = v.contiguous()
        copy_ms = median_ms(lambda: v.contiguous(), 20, 1)
        for opname, op in (("x * 2", lambda t: t * 2),
                           ("sum(dim=1)", lambda t: t.sum(dim=1))):
            nc = median_ms(lambda: op(v), 20, 1)
            co = median_ms(lambda: op(c), 20, 1)
            crow.append({"view": cname, "shape": str(tuple(v.shape)),
                         "stride": str(v.stride()), "operation": opname,
                         "non_contiguous_ms": round(nc, 4),
                         "contiguous_copy_ms": round(co, 4),
                         "slowdown": round(nc / co, 2),
                         "cost_of_contiguous()_ms": round(copy_ms, 4)})
    cdf = pd.DataFrame(crow)
    cdf.to_csv(out_path("contiguity.csv"), index=False)
    print(cdf.to_string(index=False))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    parts = sys.argv[1:] or ["verify", "moons", "pytorch"]
    if "verify" in parts:
        verify_gradients()
    if "moons" in parts:
        moons_experiment()
    if "pytorch" in parts:
        pytorch_experiments()


if __name__ == "__main__":
    main()