import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class RotaryPositionEmbedding(Function):
    """
    Rotary Position Embedding (RoPE) for attention queries and keys.

    Applies a position-dependent rotation to each pair of dimensions.
    This encodes relative position directly into attention scores without
    adding a separate positional encoding step.

    For each position pos and dimension pair index i (step 2):
        theta_i = 1 / (10000 ^ (i / head_dim))
        angle   = pos * theta_i

        out[pos, i]   = x[pos, i]   * cos(angle) - x[pos, i+1] * sin(angle)
        out[pos, i+1] = x[pos, i]   * sin(angle) + x[pos, i+1] * cos(angle)

    Applied to Q and K before the dot product. V is never rotated.
    head_dim must be even (dimensions processed in pairs).
    """

    __head_dim: int

    def __init__(self, head_dim: int):
        """
        Constructor for RotaryPositionEmbedding.

        :param head_dim: Dimension of each attention head. Must be even.
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

        theta_i = 1 / (10000 ^ (i / head_dim))

        Low dimensions (i=0) rotate fast (theta close to 1).
        High dimensions rotate slowly (theta close to 0).

        :param i: Dimension pair index (0, 2, 4, ...).
        :return: Rotation frequency theta_i.
        """
        # theta_i = 1 / (10000 ^ (i / d))
        return 1.0 / math.pow(10000.0, i / self.__head_dim)

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies RoPE to the input tensor.

        For each position pos and dimension pair (i, i+1):
            angle = pos * theta_i
            out[i]   = x[i]   * cos(angle) - x[i+1] * sin(angle)
            out[i+1] = x[i]   * sin(angle) + x[i+1] * cos(angle)

        This is a 2D rotation matrix applied to each dimension pair.

        :param tensor: Input tensor of shape (seq_len, head_dim).
        :return: Rotated tensor of same shape (seq_len, head_dim).
        """
        shape = tensor.getShape()
        seq_len = shape[0]
        head_dim = shape[1]

        values = []

        for pos in range(seq_len):
            # Step by 2 to process dimension pairs (i, i+1)
            for i in range(0, head_dim, 2):
                # theta_i = 1 / (10000 ^ (i / head_dim))
                theta = self.__getTheta(i)

                # Rotation angle depends on position
                # angle = pos * theta_i
                cos_val = math.cos(pos * theta)
                sin_val = math.sin(pos * theta)

                x0 = tensor.getValue((pos, i))      # even dimension
                x1 = tensor.getValue((pos, i + 1))  # odd dimension

                # Apply 2D rotation matrix:
                # [cos, -sin] [x0]   [x0*cos - x1*sin]
                # [sin,  cos] [x1] = [x0*sin + x1*cos]
                values.append(x0 * cos_val - x1 * sin_val)
                values.append(x0 * sin_val + x1 * cos_val)

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the gradient of RoPE (inverse rotation).

        Since rotation is orthogonal (R^T @ R = I), the backward pass
        applies the transpose rotation — same angles but sin negated:

            grad[i]   = g[i]   * cos + g[i+1] * sin
            grad[i+1] = -g[i]  * sin + g[i+1] * cos

        This is mathematically exact — no approximation needed.

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
                # Same angles as forward pass
                theta = self.__getTheta(i)
                cos_val = math.cos(pos * theta)
                sin_val = math.sin(pos * theta)

                g0 = backward.getValue((pos, i))
                g1 = backward.getValue((pos, i + 1))

                # Transpose rotation (negate sin terms):
                # [cos,  sin] [g0]   [g0*cos + g1*sin]
                # [-sin, cos] [g1] = [-g0*sin + g1*cos]
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
        :return: Newly created function node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node

    def __repr__(self) -> str:
        """
        Returns string representation of RotaryPositionEmbedding.

        :return: String representation.
        """
        return f"RotaryPositionEmbedding(head_dim={self.__head_dim})"