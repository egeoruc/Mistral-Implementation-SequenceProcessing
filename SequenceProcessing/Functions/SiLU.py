import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Function.Sigmoid import Sigmoid
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class SiLU(Function):
    """
    Sigmoid Linear Unit (SiLU) activation function.

    Used in Mistral's feed-forward network (SwiGLU variant).

    Reuses the existing Sigmoid Function from the ComputationalGraph library
    rather than reimplementing sigmoid from scratch.

    Forward pass:
        SiLU(x) = x * sigmoid(x)
        where sigmoid(x) = 1 / (1 + exp(-x))

    Derivative (by product rule):
        SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                 = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    """

    __sigmoid: Sigmoid

    def __init__(self):
        """
        Constructor for SiLU.

        Initialises the existing Sigmoid Function for internal use.
        """
        # Reuse existing Sigmoid from ComputationalGraph library
        self.__sigmoid = Sigmoid()

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies SiLU element-wise to the input tensor.

        SiLU(x) = x * sigmoid(x)

        Uses the existing Sigmoid.calculate() to compute sigmoid values,
        then multiplies element-wise with the input (hadamard product).

        :param tensor: Input tensor of shape (seq_len, d).
        :return: Output tensor after applying SiLU, same shape.
        """
        shape = tensor.getShape()

        # sigmoid(x) computed using existing Sigmoid Function
        sigmoid_values = self.__sigmoid.calculate(tensor)

        # SiLU(x) = x * sigmoid(x) — element-wise multiplication
        return tensor.hadamardProduct(sigmoid_values)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of SiLU and applies the chain rule.

        By the product rule:
            SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

        Chain rule: gradient = SiLU'(x) * upstream_gradient

        :param value: Input tensor x (pre-activation values).
        :param backward: Incoming gradient tensor from upstream.
        :return: Gradient tensor after applying the chain rule.
        """
        values = []
        shape = value.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                x = value.getValue((i, j))

                # sigmoid(x) = 1 / (1 + exp(-x))
                # Using numerically stable formulation
                if x >= 0:
                    sig = 1.0 / (1.0 + math.exp(-x))
                else:
                    exp_x = math.exp(x)
                    sig = exp_x / (1.0 + exp_x)

                # SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                grad = sig * (1.0 + x * (1.0 - sig))
                values.append(grad)

        # Apply chain rule: local_gradient * upstream_gradient (element-wise)
        return backward.hadamardProduct(Tensor(values, shape))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the edge is biased.
        :return: Newly created function node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node

    def __repr__(self) -> str:
        """
        Returns string representation of SiLU.

        :return: String representation.
        """
        return "SiLU()"