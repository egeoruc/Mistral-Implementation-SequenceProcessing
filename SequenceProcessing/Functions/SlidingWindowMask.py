from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class SlidingWindowMask(Function):
    """
    Applies a causal sliding-window mask to an attention score tensor.

    Extends the standard causal mask by adding a lookback limit W.
    Token i can only attend to tokens in range [i - W, i].

    Two conditions mask position (i, j) to -inf:
        1. j > i          — future token (standard causal mask)
        2. j < i - W      — token is older than the sliding window

    Only positions where  i - W <= j <= i  are kept.

    Example for sequence length 4, window_size=2:
        token 0: [keep, -inf, -inf, -inf]
        token 1: [keep, keep, -inf, -inf]
        token 2: [-inf, keep, keep, -inf]
        token 3: [-inf, -inf, keep, keep]
    """

    __window_size: int

    def __init__(self, window_size: int):
        """
        Constructor for SlidingWindowMask.

        :param window_size: Maximum number of past tokens each position
                            can attend to. Must be >= 1.
        """
        if window_size < 1:
            raise ValueError(
                f"window_size must be >= 1, got {window_size}."
            )
        self.__window_size = window_size

    def getWindowSize(self) -> int:
        """
        Getter for window size.

        :return: Sliding window size.
        """
        return self.__window_size

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies the sliding-window causal mask to the attention score tensor.

        Masking rule for position (i, j):
            if j > i:          set to -inf  (future token — causal mask)
            if j < i - W:      set to -inf  (outside sliding window)
            otherwise:         keep original value

        After softmax, -inf positions become exactly 0 and are ignored.

        :param tensor: Input attention score tensor of shape (seq_len, seq_len).
        :return: Masked tensor of the same shape.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                if j > i:
                    # Future position — standard causal mask condition
                    values.append(float("-inf"))
                elif j < i - self.__window_size:
                    # Outside sliding window — too far in the past
                    # Condition: j < i - W
                    values.append(float("-inf"))
                else:
                    # Within causal sliding window [i-W, i] — keep score
                    values.append(tensor.getValue((i, j)))

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the sliding-window mask.

        The mask is a hard gate: -inf positions become 0 after softmax
        so their upstream gradient is already 0. For kept positions the
        mask is the identity function so the gradient passes through unchanged.

        Mathematically: d/dx(mask(x)) = 1 for kept positions, 0 for masked.
        Multiplying backward by all-ones returns backward unchanged.

        :param value: Current tensor value (post-mask).
        :param backward: Incoming gradient tensor from upstream.
        :return: Gradient tensor (backward passed through unchanged).
        """
        shape = value.getShape()

        # Identity gradient: mask kept positions have derivative 1
        # Masked positions already have 0 upstream gradient from softmax
        ones = [1.0] * (shape[0] * shape[1])
        return backward.hadamardProduct(Tensor(ones, shape))

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
        Returns string representation of SlidingWindowMask.

        :return: String representation.
        """
        return f"SlidingWindowMask(window_size={self.__window_size})"