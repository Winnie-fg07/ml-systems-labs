# Lab 1: Environment and first system measurements

## Goal
The goal of this lab was to set up a reproducible Python 3.11.8 environment, train two baseline models (logistic regression and random forest) on the Breast Cancer Wisconsin dataset, and measure their training time, inference latency, model size and memory use. I then compared the measurements with Cloud, Edge, Mobile and TinyML budgets to decide where each model could be deployed.

## Method
- Python 3.11.8 in a virtual environment, with packages pinned in requirements.txt.
- I used pyarrow==15.0.2 instead of 16.1.0 because mlflow==2.14.1 requires pyarrow<16.
- Dataset: Breast Cancer Wisconsin (Diagnostic), 70/30 stratified split, random_state=42.
- Models: LogisticRegression (max_iter=1000) and RandomForest (100 trees).
- Training time: 1 warm-up run, then the median of 5 runs.
- Inference latency: median of 100 single-sample predictions after 1 warm-up.
- Model size: saved with joblib.dump, one file per model (the lab has two models).
- Memory: measured with memory_profiler for the whole Python process, so it includes numpy and scikit-learn, not only the model.

## Results

| Metric | LogisticRegression | RandomForest |
|---|---|---|
| Test accuracy | 0.9474 | 0.9357 |
| Training time (median) | 0.6295 s | 0.2381 s |
| Inference latency (median) | 0.1378 ms | 5.8629 ms |
| Model size | 1055 bytes (1.03 KB) | 290889 bytes (284.07 KB) |
| Peak memory, training | 177.8 MiB | 179.5 MiB |
| Peak memory, inference | 177.9 MiB | 179.5 MiB |

### Budgets

| | Memory | Latency | Model size |
|---|---|---|---|
| Cloud | ≥ 1 GB | ≤ 100 ms | ≤ 500 MB |
| Edge | 256 to 1024 MB | ≤ 50 ms | ≤ 50 MB |
| Mobile | 64 to 256 MB | ≤ 20 ms | ≤ 10 MB |
| TinyML | ≤ 256 KB | ≤ 10 ms | ≤ 100 KB |

### LogisticRegression fit (178 MiB, 0.14 ms, 1.03 KB)

| | Memory | Latency | Size |
|---|---|---|---|
| Cloud | pass | pass | pass |
| Edge | pass | pass | pass |
| Mobile | pass | pass | pass |
| TinyML | fail | pass | pass |

### RandomForest fit (180 MiB, 5.86 ms, 284 KB)

| | Memory | Latency | Size |
|---|---|---|---|
| Cloud | pass | pass | pass |
| Edge | pass | pass | pass |
| Mobile | pass | pass | pass |
| TinyML | fail | pass | fail |

### Deployment argument
Both models fit the Cloud, Edge and Mobile budgets on memory, latency and size. Neither fits TinyML on memory, because the whole Python process uses about 178 MiB while the limit is 256 KB. LogisticRegression (1.03 KB, 0.14 ms) is within the TinyML limits for size and latency, so with a runtime other than Python it could plausibly run there. RandomForest fails TinyML on size as well (284 KB against a 100 KB limit), although its 5.86 ms latency is still under the 10 ms limit.

## Three conclusions
1. Accuracy against cost: The simpler model won on every measure here: LogisticRegression is slightly more accurate (0.9474 against 0.9357), far smaller, and far faster per prediction.
2. Training against inference: A random forest has to run the sample through all 100 trees and combine their votes, and scikit-learn adds fixed overhead per call, which is why one prediction takes 5.86 ms. Logistic regression predicts with a single dot product, so it takes 0.14 ms. Training goes the other way here, probably because logistic regression is solved iteratively (up to 1000 iterations) on unscaled features, while the trees are built quickly on a small dataset.
3. Process memory against model size: Both models use about 178 to 180 MiB even though their files are 1 KB and 284 KB, so almost all of that memory is the Python process itself: the interpreter, numpy, scipy and scikit-learn being loaded, not the model. This shows why TinyML fails on memory for both models: a 256 KB budget can’t hold a Python runtime at all.