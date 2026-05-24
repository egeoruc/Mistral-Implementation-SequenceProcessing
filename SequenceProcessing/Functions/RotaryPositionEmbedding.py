import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class RotaryPositionEmbedding(Function):
    """
    Rotary Position Embedding (RoPE) for attention queries and keys.

    Applies a position-dependent rotation to each pair of dimensions
    in the input tensor. This encodes relative position information
    directly into the attention scores without adding a separate
    positional encoding step.

    For each position pos and dimension pair (2i, 2i+1):

        theta_i = 1 / (10000 ^ (2i / head_dim))

        out[pos, 2i]   = x[pos, 2i]   * cos(pos * theta_i)
                       - x[pos, 2i+1] * sin(pos * theta_i)

        out[pos, 2i+1] = x[pos, 2i]   * sin(pos * theta_i)
                       + x[pos, 2i+1] * cos(pos * theta_i)

    Applied to Q and K before the dot product in attention.
    V is never rotated.

    Input shape:  (seq_len, head_dim)
    Output shape: (seq_len, head_dim)   -- same shape

    head_dim must be even.
    """

    __head_dim: int

    def __init__(self, head_dim: int):
        """
        Constructor for RotaryPositionEmbedding.

        :param head_dim: Dimension of each attention head.
                         Must be even (pairs of dimensions are rotated together).
        """
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE, got {head_dim}."
            )
        self.__head_dim = head_dim

    def getHeadDim(self) -> int:
        """
        Getter for head dimension.

        :return: Head dimension.
        """
        return self.__head_dim

    def __getTheta(self, i: int) -> float:
        """
        Computes the rotation frequency for dimension pair i.

        theta_i = 1 / (10000 ^ (2i / head_dim))

        :param i: Dimension pair index (0-based, step 2).
        :return: Rotation frequency theta_i.
        """
        return 1.0 / math.pow(10000.0, i / self.__head_dim)

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies RoPE to the input tensor.

        Each pair of dimensions (2i, 2i+1) at each position pos
        is rotated by angle pos * theta_i.

        :param tensor: Input tensor of shape (seq_len, head_dim).
        :return: Rotated tensor of same shape (seq_len, head_dim).
        """
        shape = tensor.getShape()
        seq_len = shape[0]
        head_dim = shape[1]

        values = []

        for pos in range(seq_len):
            for i in range(0, head_dim, 2):
                theta = self.__getTheta(i)
                cos_val = math.cos(pos * theta)
                sin_val = math.sin(pos * theta)

                x0 = tensor.getValue((pos, i))
                x1 = tensor.getValue((pos, i + 1))

                # Rotate the pair
                values.append(x0 * cos_val - x1 * sin_val)
                values.append(x0 * sin_val + x1 * cos_val)

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the gradient of RoPE.

        Since rotation is an orthogonal transformation (R^T @ R = I),
        the backward pass applies the transpose rotation — which is
        the same rotation with the sin terms negated.

        For gradient g at output (pos, 2i) and (pos, 2i+1):

            grad[pos, 2i]   = g[pos, 2i]   * cos - g[pos, 2i+1] * (-sin)
                            = g[pos, 2i]   * cos + g[pos, 2i+1] * sin

            grad[pos, 2i+1] = g[pos, 2i]   * (-sin) + g[pos, 2i+1] * cos
                            = -g[pos, 2i]  * sin  + g[pos, 2i+1] * cos

        :param value: Forward output tensor of shape (seq_len, head_dim).
        :param backward: Incoming gradient tensor of shape (seq_len, head_dim).
        :return: Gradient tensor of shape (seq_len, head_dim).
        """
        shape = value.getShape()
        seq_len = shape[0]
        head_dim = shape[1]

        values = []

        for pos in range(seq_len):
            for i in range(0, head_dim, 2):
                theta = self.__getTheta(i)
                cos_val = math.cos(pos * theta)
                sin_val = math.sin(pos * theta)

                g0 = backward.getValue((pos, i))
                g1 = backward.getValue((pos, i + 1))

                # Transpose rotation
                values.append(g0 * cos_val + g1 * sin_val)
                values.append(-g0 * sin_val + g1 * cos_val)

        return Tensor(values, shape)

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the edge is biased.
        :return: Newly created computational node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node