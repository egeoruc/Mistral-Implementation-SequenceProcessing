from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class SlidingWindowMask(Function):
   

    __window_size: int

    def __init__(self, window_size: int):
      
        if window_size < 1:
            raise ValueError(
                f"window_size must be >= 1, got {window_size}."
            )
        self.__window_size = window_size

    def getWindowSize(self) -> int:
        
        return self.__window_size

    def calculate(self, tensor: Tensor) -> Tensor:
        
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                if j > i:
                   
                    values.append(float("-inf"))
                elif j < i - self.__window_size:
                   
                    values.append(float("-inf"))
                else:
                   
                    values.append(tensor.getValue((i, j)))

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        values = []
        shape = value.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(1.0)

        return backward.hadamardProduct(Tensor(values, shape))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node