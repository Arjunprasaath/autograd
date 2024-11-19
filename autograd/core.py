import math
import random
from graphviz import Digraph

class tensor:
    def __init__(self, data, _parent = (), _op = ''):
        self.data = float(data)
        self.grad = 0
        self._parent = set(_parent)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"tensor(data = {self.data}, grad = {self.grad})" # ,  parent = {self._parent}, operation = {self._op})"

    def __add__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        output = tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        output._backward = _backward
        return output
    
    def __sub__(self, other):
        other  = other if isinstance(other, tensor) else tensor(other)
        output = tensor(self.data + ( - other.data), (self, other), '-')

        def _backward():
            self.grad += output.grad
            other.grad -= output.grad
        output._backward = _backward
        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other)
        output = tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = _backward
        return output
    
    def __truediv__(self, other):
        other  = other if isinstance(other, tensor) else tensor(other)
        output = tensor(self.data / other.data, (self, other), '/')

        def _backward():
            if isinstance(other, tensor):
                self.grad += (1 / other.data) * output.grad
                other.grad -= (self.data / other.data ** 2) * output.grad
            else:
                self.grad += (1 / other) *  output.grad
        output._backward = _backward
        return output            
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        output = tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * output.grad
        output._backward = _backward
        return output
    
    def __neg__(self):
        output = tensor(self.data * -1, (self,), '~')

        def _backward():
            self.gard += (-1) * output.gard
        output._backward = _backward
        return output
    
    # Reverse functions
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __rtruediv__(self, other):
        return (tensor(other) / self)
    
    # Activation functions
    def sigmoid(self):
        sig = 1 / (1 + math.exp(-self.data))
        output = tensor(sig, (self, ), '1 / (1 + exp(-x))')

        def _backward():
            self.grad += (sig * (1 - sig)) * output.grad
        output._backward = _backward
        return output
    
    def tanh(self):
        th = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        output = tensor(th, (self, ), '(exp(2 * x) - 1) / (exp(2 * x) + 1)')
        
        def _backward():
            self.grad += (1 - th ** 2) * output.grad
        output._backward = _backward
        return output
    
    def relu(self):
        r = 0 if self.data < 0 else self.data
        output = tensor(r, (self,), 'max(0, x)')
        
        def _backward():
            self.grad += (1 if r > 0 else 0) * output.grad
        output._backward = _backward
        return output

    # Calculating gradient
    def backward(self):
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._parent:
                    build_topo(parent)
                topo.append(node)
        build_topo(self)
        
        self.grad = 1
        for p in reversed(topo):
            p._backward()

    # Visualizing helper
    def trace(self):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._parent:
                    edges.add((child, v))
                    build(child)
        build(self)
        return nodes, edges

    # Visualizing tensor
    def visualize_tensor(self, format='svg', rankdir='LR'):
        """
        format: png | svg | ...
        rankdir: TB (top to bottom graph) | LR (left to right)
        """
        assert rankdir in ['LR', 'TB']
        nodes, edges = self.trace()
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
        
        for n in nodes:
            dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))
        
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
        return dot
