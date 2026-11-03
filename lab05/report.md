# Lab 5 – Building an Autograd Engine and Exploring PyTorch Performance

## 1. Goal

The goal of this laboratory was to understand how automatic differentiation (autograd) works and how gradients are calculated using computational graphs. I implemented a simple scalar-based autograd engine in Python and compared its gradient calculations with PyTorch and numerical gradients.

The laboratory also investigated neural network training performance, PyTorch execution modes, tensor properties, and how memory layout affects computational speed.

## 2. Method

### Task 1–3: Building the Autograd Engine

I implemented a custom autograd engine using a `Value` class. Each Value stores:
- `data`: the numerical value.
- `grad`: the gradient calculated during backpropagation.
- `parents`: the previous nodes used to calculate the current value.
- `backward`: the local derivative rule for each operation.

The engine supports addition, multiplication, power and ReLU operations. Additional sigmoid and logarithm operations were implemented for the binary cross-entropy loss used in neural network training.

The backward pass follows three main steps:
1. A topological sort is performed to organise the computational graph.
2. The output gradient is initialised to 1.
3. Gradients are propagated backwards through the graph, accumulating contributions using `+=`.

### Task 4: Gradient Verification

I compared the gradients produced by my custom engine with PyTorch autograd and numerical gradients calculated using the central difference method.

I also tested gradient accumulation using the expression:

`y = x * x + x`

At x = 3, the expected gradient is 7. The engine successfully produced this result.

### Task 5: Neural Network Training

I trained a small neural network using the make_moons dataset and compared my custom engine with PyTorch.

The model used:
- Architecture: 2 → 8 → 1.
- ReLU activation in the hidden layer.
- Sigmoid activation and binary cross-entropy loss.
- Full-batch stochastic gradient descent.
- Learning rate: 0.5.
- Epochs: 100.
- Train/test split: 70% / 30%, stratified.
- Random seed: 42.

Both implementations started with the same weights and used the same training settings. Five runs were performed after a warm-up, and median execution times were compared.

### Task 6: Exploring PyTorch

I investigated three areas:

**Autograd graph:** Examined the computational graph created by PyTorch to understand how operations and gradients are connected.

**Execution performance:** Compared eager execution, JIT tracing and torch.compile. The measurements used 20 warm-up calls, followed by 100 runs for batch size 1 and 20 runs for batch size 128.

**Tensor contiguity:** Compared operations on non-contiguous tensor views and their contiguous copies. I examined tensor shapes, strides and execution times for multiplication and summation.

## 3. Results

### 3.1 Gradient Verification

The custom engine produced the same gradients as PyTorch. The numerical gradient comparison also showed extremely small differences.

| Comparison | Maximum absolute difference | Result |
|---|---:|---|
| Engine vs PyTorch | 0.0 | PASS |
| Engine vs Numerical | 6.4062 × 10⁻¹⁰ | PASS |
| PyTorch vs Numerical | 6.4062 × 10⁻¹⁰ | PASS |

The gradient accumulation test also passed, producing the expected gradient of 7.

### 3.2 Neural Network Performance: My Engine vs PyTorch

| Implementation | Median time (s) | Final training loss | Test accuracy |
|---|---:|---:|---:|
| My engine | 73.7175 | 0.235766 | 0.92 |
| PyTorch | 0.3646 | 0.235766 | 0.92 |

Both implementations achieved identical final training loss and test accuracy. However, my custom engine was approximately 202 times slower than PyTorch.

This difference is expected because my engine performs calculations using individual Python scalar operations, whereas PyTorch uses optimised tensor operations.

### 3.3 Eager vs JIT Tracing vs torch.compile

**Batch size 1**

| Execution mode | Median time (ms) | Speedup vs eager |
|---|---:|---:|
| Eager | 0.4011 | 1.000× |
| JIT trace | 0.4114 | 0.975× |
| torch.compile | 0.3815 | 1.051× |

**Batch size 128**

| Execution mode | Median time (ms) | Speedup vs eager |
|---|---:|---:|
| Eager | 0.8873 | 1.000× |
| JIT trace | 0.8576 | 1.035× |
| torch.compile | 0.6845 | 1.296× |

For batch size 1, torch.compile provided a small improvement of approximately 5.1%. For batch size 128, it provided a larger improvement of approximately 29.6%.

JIT tracing was slightly slower for batch size 1 but slightly faster for batch size 128.

### 3.4 Tensor Properties

The following tensor properties were observed:

| Tensor | Shape | Stride | Contiguous |
|---|---|---|---|
| Input x | (1, 1, 28, 28) | (784, 784, 28, 1) | True |
| Linear weight | (256, 784) | (784, 1) | True |
| Linear weight transpose | (784, 256) | (1, 784) | False |
| Hidden activation | (1, 256) | (256, 1) | True |

The transposed linear weight was non-contiguous because transposing changes how the tensor elements are accessed in memory without necessarily copying the underlying data.

### 3.5 Contiguity Performance

| Tensor view | Operation | Non-contiguous (ms) | Contiguous (ms) | Slowdown |
|---|---|---:|---:|---:|
| Transpose | x * 2 | 1.7943 | 1.8706 | 0.96× |
| Transpose | sum(dim=1) | 3.5484 | 1.4502 | 2.45× |
| Strided slice | x * 2 | 1.1413 | 0.7464 | 1.53× |
| Strided slice | sum(dim=1) | 0.6880 | 0.8254 | 0.83× |

The cost of creating a contiguous copy was:

| Tensor view | contiguous() cost (ms) |
|---|---:|
| Transpose | 8.1207 |
| Strided slice | 1.1794 |

The results show that non-contiguous tensors do not always perform worse. The transpose was slower for summation but slightly faster for multiplication after comparison with the contiguous tensor. Similarly, the strided slice was slower for multiplication but faster for summation.

Creating a contiguous copy also introduces an additional cost that should be considered.

### 3.6 Autograd Graph

The PyTorch computational graph was inspected, starting from `SumBackward0`. It contained operations such as `AddmmBackward0`, `ReluBackward0` and `TBackward0`, as well as gradient accumulation nodes.

This demonstrates how PyTorch records the operations performed during the forward pass and uses the graph to calculate gradients during backpropagation.

## 4. When Compilation Helps

Compilation can improve performance by reducing Python overhead, optimising operations and combining smaller operations where possible.

It is particularly useful when the same model and input shapes are executed repeatedly because the initial compilation cost can be distributed across multiple runs.

However, compilation may provide little improvement for very small models, single executions or models already dominated by large matrix operations.

My results support this observation. For batch size 1, torch.compile improved execution speed by only 5.1%, whereas for batch size 128, the improvement was 29.6%.

This suggests that compilation provided a greater performance benefit when processing a larger batch of inputs.

## 5. Three Conclusions

1. **Gradient accuracy:** I successfully implemented a basic autograd engine that calculated gradients correctly. The maximum difference between my engine and PyTorch was 0.0, while the numerical comparison produced a very small difference of approximately 6.41 × 10⁻¹⁰. This confirmed that my backward propagation and gradient accumulation worked correctly.

2. **Training performance:** My custom engine and PyTorch achieved the same final training loss of 0.235766 and test accuracy of 92%. However, my engine took 73.7175 seconds compared with PyTorch's 0.3646 seconds. This demonstrated the performance advantage of PyTorch's optimised tensor operations over a scalar-based Python implementation.

3. **Execution and memory optimisation:** The experiments showed that compilation and tensor contiguity can affect performance depending on the workload. torch.compile achieved its best measured improvement with batch size 128, reaching a 1.296× speedup. The contiguity experiments also showed that a non-contiguous tensor is not necessarily slower for every operation.

Overall, this laboratory helped me understand the principles behind automatic differentiation, computational graphs, gradient verification and performance optimisation in PyTorch.