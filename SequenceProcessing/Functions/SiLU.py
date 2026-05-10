import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class SiLU(Function):
    """
    Used in Mistral's feed-forward network (SwiGLU variant).

    Forward pass:
        SiLU(x) = x * sigmoid(x)

    Derivative (by product rule):
        SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """

    def __init__(self):
       
        pass

    def __sigmoid(self, x: float) -> float:
    
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            exp_x = math.exp(x)
            return exp_x / (1.0 + exp_x)

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies SiLU element-wise to the input tensor.

        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = tensor.getValue((i, j))
                values.append(x * self.__sigmoid(x))

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of SiLU and applies the chain rule.

        SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

        """
        values = []
        shape = value.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = value.getValue((i, j))
                sig = self.__sigmoid(x)
                grad = sig * (1.0 + x * (1.0 - sig))
                values.append(grad)

        return backward.hadamardProduct(Tensor(values, shape))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:

        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node