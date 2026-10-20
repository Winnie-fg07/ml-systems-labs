import os, random, time, statistics, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile

SEED = 42
BATCH = 128
EPOCHS = 5
LR = 1e-3

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "..", "data")


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ---------- models ----------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(49, 64)
        self.pos = nn.Parameter(torch.randn(1, 16, 64) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        b = x.shape[0]
        # 28x28 image -> 16 patches of 7x7, each flattened to 49 values
        x = x.reshape(b, 4, 7, 4, 7).permute(0, 1, 3, 2, 4).reshape(b, 16, 49)
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


# ---------- hand-calculated FLOPs (Ch. 4 formulas, 2 FLOPs per multiply-add) ----------
def lin_flops(i, o):
    return 2 * i * o


def conv_flops(k, ci, co, h, w):
    return 2 * k * k * ci * co * h * w


def attn_layer_flops(seq, d, ff):
    return {
        "qkv_projections": 2 * seq * d * 3 * d,
        "attention_scores": 2 * seq * seq * d,
        "weighted_sum_of_values": 2 * seq * seq * d,
        "output_projection": 2 * seq * d * d,
        "feed_forward": 2 * seq * d * ff + 2 * seq * ff * d,
    }


def hand_totals():
    mlp = lin_flops(784, 256) + lin_flops(256, 128) + lin_flops(128, 10)
    cnn = (conv_flops(3, 1, 16, 28, 28) + conv_flops(3, 16, 32, 14, 14)
           + lin_flops(32 * 7 * 7, 10))
    att = (2 * 16 * 49 * 64 + 2 * sum(attn_layer_flops(16, 64, 128).values())
           + lin_flops(64, 10))
    return {"MLP": mlp, "CNN": cnn, "Attention": att}


# ---------- one-layer comparison: hand vs thop ----------
def thop_macs(module, x):
    m = copy.deepcopy(module).eval()
    macs, _ = profile(m, inputs=(x,), verbose=False)
    return macs


def layer_comparison():
    rows = []
    # Linear 784 -> 256
    hand = lin_flops(784, 256)
    macs = thop_macs(nn.Linear(784, 256), torch.randn(1, 784))
    rows.append(("Linear 784->256", hand, macs))
    # Conv2d(16, 32, 3, padding=1) on a 14x14 feature map
    hand = conv_flops(3, 16, 32, 14, 14)
    macs = thop_macs(nn.Conv2d(16, 32, 3, padding=1), torch.randn(1, 16, 14, 14))
    rows.append(("Conv2d 16->32, 3x3, 14x14", hand, macs))
    # One Transformer encoder layer, seq 16, d_model 64
    hand = sum(attn_layer_flops(16, 64, 128).values())
    layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
    macs = thop_macs(layer, torch.randn(1, 16, 64))
    rows.append(("Transformer encoder layer (seq 16)", hand, macs))
    df = pd.DataFrame(rows, columns=["layer", "hand_flops", "thop_macs"])
    df["thop_as_flops_(x2)"] = df["thop_macs"] * 2
    df["hand_over_thop_flops"] = (df["hand_flops"] / df["thop_as_flops_(x2)"]).round(3)
    return df


# ---------- data ----------
def get_loaders():
    tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full = datasets.FashionMNIST(root=DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root=DATA, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full, [55000, 5000], generator=g)
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=BATCH)
    test_loader = DataLoader(test, batch_size=BATCH)
    return train_loader, val_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            n += len(y)
    return correct / n


def train_model(model, name, train_loader, val_loader):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    log = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            count += len(y)
        val_acc = evaluate(model, val_loader)
        log.append({"model": name, "epoch": epoch,
                    "train_loss": total / count, "val_acc": val_acc})
        print(f"{name} epoch {epoch}/{EPOCHS} "
              f"train_loss={total / count:.4f} val_acc={val_acc:.4f}")
    return log


def latency_ms(model, batch, runs):
    model.eval()
    x = torch.randn(batch, 1, 28, 28)
    times = []
    with torch.no_grad():
        model(x)  # one warm-up
        for _ in range(runs):
            t = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    set_seed()
    print("torch threads:", torch.get_num_threads())

    print("\n=== One-layer FLOPs: hand vs thop ===")
    lc = layer_comparison()
    print(lc.to_string(index=False))
    lc.to_csv(os.path.join(RESULTS, "layer_flops.csv"), index=False)

    train_loader, val_loader, test_loader = get_loaders()
    makers = {"MLP": MLP, "CNN": CNN, "Attention": PatchAttention}
    hand = hand_totals()
    rows, logs = [], []

    for name, make in makers.items():
        print(f"\n=== {name} ===")
        set_seed()
        model = make()
        info = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        print(info)
        with open(os.path.join(RESULTS, f"summary_{name}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(str(info))

        macs = thop_macs(model, torch.randn(1, 1, 28, 28))
        logs += train_model(model, name, train_loader, val_loader)
        test_acc = evaluate(model, test_loader)
        lat1 = latency_ms(model, 1, 100)
        lat128 = latency_ms(model, 128, 20)
        rows.append({
            "model": name,
            "params": info.total_params,
            "hand_flops": hand[name],
            "thop_macs": int(macs),
            "thop_flops_x2": int(macs * 2),
            "latency_b1_ms": round(lat1, 4),
            "latency_b128_ms": round(lat128, 4),
            "test_acc": round(test_acc, 4),
        })
        print(rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, "results.csv"), index=False)
    pd.DataFrame(logs).to_csv(os.path.join(RESULTS, "training_log.csv"), index=False)
    print("\n=== Summary ===")
    print(res.to_string(index=False))

    # trade-off plots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for _, r in res.iterrows():
        axes[0].scatter(r["params"], r["test_acc"], s=60)
        axes[0].annotate(r["model"], (r["params"], r["test_acc"]),
                         textcoords="offset points", xytext=(6, 4))
        axes[1].scatter(r["latency_b1_ms"], r["test_acc"], marker="o", s=60,
                        label=f"{r['model']} batch 1")
        axes[1].scatter(r["latency_b128_ms"], r["test_acc"], marker="s", s=60,
                        label=f"{r['model']} batch 128")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Parameters (log scale)")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Accuracy vs parameters")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Median latency (ms, log scale)")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Accuracy vs latency (circle: batch 1, square: batch 128)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "tradeoffs.png"), dpi=150)


if __name__ == "__main__":
    main()