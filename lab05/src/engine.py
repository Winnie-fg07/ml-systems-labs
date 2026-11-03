import math


class Value:
    """A scalar that remembers how it was computed (a node in the graph)."""

    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(parents)   # nodes this one was computed from
        self._op = op                    # name of the operation, for inspection
        self._backward = lambda: None    # local derivative rule, set by each operation

    def __repr__(self):
        return f"Value(data={self.data:.6f}, grad={self.grad:.6f})"

    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Value) else Value(x)

    # ---------- operations: each one builds a graph node during the forward pass ----------
    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only int or float powers are supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # extra operations needed for the sigmoid + BCE model in task 5
    def sigmoid(self):
        x = self.data
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def log(self):
        x = max(self.data, 1e-12)        # avoids log(0)
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += (1.0 / x) * out.grad
        out._backward = _backward
        return out

    # convenience operators built from the basic ones
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1

    def __rtruediv__(self, other):
        return self._wrap(other) * self ** -1

    # ---------- backward pass ----------
    def backward(self):
        # 1. topological sort (iterative, so long graphs do not hit Python's recursion limit)
        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if id(parent) not in visited:
                    stack.append((parent, False))
        # 2. output gradient is 1
        self.grad = 1.0
        # 3. visit nodes in reverse order; each rule ACCUMULATES with +=
        for node in reversed(topo):
            node._backward()