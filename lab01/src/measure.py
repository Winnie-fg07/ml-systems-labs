import os, random, time, statistics
import numpy as np
import pandas as pd
import joblib
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    random.seed(42)
    np.random.seed(42)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    factories = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    }

    acc_rows, cost_rows = [], []

    for name, make in factories.items():
        def fit_once():
            m = make()
            m.fit(X_train, y_train)
            return m

        # training time: 1 warm-up, then 5 timed runs, take the median
        fit_once()
        train_times = []
        for _ in range(5):
            t = time.perf_counter()
            fit_once()
            train_times.append(time.perf_counter() - t)
        train_median = statistics.median(train_times)

        model = fit_once()
        acc = accuracy_score(y_test, model.predict(X_test))
        acc_rows.append({"model": name, "test_accuracy": round(acc, 4)})

        # single-sample inference: 1 warm-up, then 100 runs, median in ms
        sample = X_test[:1]
        model.predict(sample)
        infer_times = []
        for _ in range(100):
            t = time.perf_counter()
            model.predict(sample)
            infer_times.append((time.perf_counter() - t) * 1000)
        infer_median_ms = statistics.median(infer_times)

        # model size
        path = os.path.join(results_dir, f"model_{name}.joblib")
        joblib.dump(model, path)
        size_bytes = os.path.getsize(path)

        # peak memory (MiB) during training and during inference
        train_peak = float(np.max(memory_usage((fit_once, (), {}), interval=0.01, max_usage=True)))

        def infer_loop():
            for _ in range(100):
                model.predict(sample)

        infer_peak = float(np.max(memory_usage((infer_loop, (), {}), interval=0.01, max_usage=True)))

        cost_rows.append({
            "model": name,
            "train_time_median_s": round(train_median, 4),
            "infer_latency_median_ms": round(infer_median_ms, 4),
            "size_bytes": size_bytes,
            "size_KB": round(size_bytes / 1024, 2),
            "train_peak_mem_MiB": round(train_peak, 1),
            "infer_peak_mem_MiB": round(infer_peak, 1),
        })

    pd.DataFrame(acc_rows).to_csv(os.path.join(results_dir, "baseline_accuracy.csv"), index=False)
    pd.DataFrame(cost_rows).to_csv(os.path.join(results_dir, "system_cost.csv"), index=False)
    print(pd.DataFrame(acc_rows))
    print(pd.DataFrame(cost_rows).T)


if __name__ == "__main__":
    main()