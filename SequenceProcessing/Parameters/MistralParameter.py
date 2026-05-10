from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Initialization.Initialization import Initialization
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Optimizer.Optimizer import Optimizer


class MistralParameter(NeuralNetworkParameter):
    """
    Parameter class for the simplified Mistral-like model.

    Stores all hyperparameters needed to construct and train a MistralModel,
    following the same style as TransformerParameter.
    """

    __d_model: int       
    __n_heads: int       
    __n_kv_heads: int     
    __n_layers: int       
    __ffn_dim: int        
    __window_size: int    
    __vocab_size: int     
    __epsilon: float      

    def __init__(self,
                 seed: int,
                 epoch: int,
                 optimizer: Optimizer,
                 initialization: Initialization,
                 loss: Function,
                 d_model: int,
                 n_heads: int,
                 n_kv_heads: int,
                 n_layers: int,
                 ffn_dim: int,
                 window_size: int,
                 vocab_size: int,
                 epsilon: float):
        """
        Constructor for MistralParameter.

        :param seed: Random seed for reproducibility.
        :param epoch: Number of training epochs.
        :param optimizer: Optimization algorithm (e.g. AdamW).
        :param initialization: Weight initialization method (e.g. RandomInitialization).
        :param loss: Loss function (e.g. CrossEntropyLoss).
        :param d_model: Embedding and hidden dimension. Must be divisible by n_heads.
        :param n_heads: Number of query attention heads.
        :param n_kv_heads: Number of key/value heads for grouped-query attention.
                           Must satisfy: n_kv_heads <= n_heads and
                           n_heads % n_kv_heads == 0.
        :param n_layers: Number of stacked Mistral transformer blocks.
        :param ffn_dim: Inner (hidden) dimension of the feed-forward network.
        :param window_size: Sliding-window attention size. Each token attends
                            only to the previous window_size tokens.
        :param vocab_size: Vocabulary / output class count.
        :param epsilon: Small constant added inside RMSNorm for numerical stability.
        """
        super().__init__(seed, epoch, optimizer, initialization, loss, 0.0, 1)

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})."
            )

        self.__d_model = d_model
        self.__n_heads = n_heads
        self.__n_kv_heads = n_kv_heads
        self.__n_layers = n_layers
        self.__ffn_dim = ffn_dim
        self.__window_size = window_size
        self.__vocab_size = vocab_size
        self.__epsilon = epsilon

    def getDModel(self) -> int:
       
        return self.__d_model

    def getNHeads(self) -> int:
       
        return self.__n_heads

    def getNKVHeads(self) -> int:
       
        return self.__n_kv_heads

    def getNLayers(self) -> int:
        
        return self.__n_layers

    def getFFNDim(self) -> int:
        
        return self.__ffn_dim

    def getWindowSize(self) -> int:
      
        return self.__window_size

    def getVocabSize(self) -> int:
        
        return self.__vocab_size

    def getEpsilon(self) -> float:
        
        return self.__epsilon

    def getHeadDim(self) -> int:
        
        return self.__d_model // self.__n_heads

    def getKVHeadDim(self) -> int:
       
        return self.__d_model // self.__n_heads

    def getGroupSize(self) -> int:
       
        return self.__n_heads // self.__n_kv_heads