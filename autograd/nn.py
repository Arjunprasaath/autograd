import random
from autograd.core import tensor
from graphviz import Digraph

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
    
class Neuron(Module):
    def __init__(self, n_input, nonlin = ''):
        """
        Build Neurons based on number of input dimensions

        n_input is the number of input dimensions
        """
        self.w = [tensor(random.uniform(-1, 1)) for _ in range(n_input)]
        self.b = tensor(1.0)
        self.nonlin = nonlin

    def __repr__(self):
        return f"Weights: {self.w}, \nBias: {self.b} \nNon Linearity: {self.nonlin}"
    
    def parameters(self):
        return self.w + [self.b]
    
    def __call__(self, x):
        output = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == "ReLU":
            output = output.relu()
        elif self.nonlin == "Tanh":
            output = output.tanh()
        elif self.nonlin == "Sigmoid":
            output = output.sigmoid()
        return output
    

class Layer(Module):
    def __init__(self, n_input, n_output, **kwargs):
        """
        Build a Layer of Neurons based on n_output

        n_input is the number input dimensions
        n_output is the number of Neurons in one Layer
        """
        self.Layer = [Neuron(n_input, **kwargs) for _ in range(n_output)]

    def __repr__(self):
        return f"Layers: {self.Layer} \n"
    
    def parameters(self):
        return [p for n in self.Layer for p in n.parameters()]

    def __call__(self, x):
        logits = [n(x) for n in self.Layer]
        return logits

class MLP(Module):
    def __init__(self, n_input, n_outputs, non_linearity):
        """
        Build layers of Neurons based on list of n_outputs

        n_input is the number of input dimensions
        n_outputs is the list of number of Neurons in each Layer
        non_linearity is the type of activation function to be used
        """
        self.last_input, self.last_output = None, None
        size = [n_input] + n_outputs
        self.layers = [Layer(size[i], size[i + 1], nonlin = non_linearity) for i in range(len(n_outputs))]

    def __repr__(self):
        return f"FFN: {self.layers}"
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
    
    def __call__(self, x):
        self.last_input = x
        for ll in self.layers:
            x = ll(x)
        self.last_output = x
        return x

    def trace(self):
        """Trace the computation graph from the last output tensors."""
        if not self.last_output:
            return set(), set()
        outputs = self.last_output if isinstance(self.last_output, list) else [self.last_output]
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for parent in v._parent:
                    edges.add((parent, v))
                    build(parent)

        for output in outputs:
            build(output)
        return nodes, edges

    def draw_dot(self, format='svg', rankdir='LR'):
        nodes, edges = self.trace()
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

        for n in nodes:
            dot.node(name=str(id(n)), label="{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot

    def visualize(self, format='svg', rankdir='LR'):
        """Visualize the computation graph of the latest forward pass."""
        return self.draw_dot(format, rankdir)
