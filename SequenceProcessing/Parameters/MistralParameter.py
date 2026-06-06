from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Initialization.Initialization import Initialization
from ComputationalGraph.Optimizer.Optimizer import Optimizer

from SequenceProcessing.Parameters.TransformerParameter import TransformerParameter


class MistralParameter(TransformerParameter):
    """
    Parameter class for the simplified Mistral-like model.

    Extends TransformerParameter and adds Mistral-specific parameters:
        - n_kv_heads: number of key/value heads for GQA
        - ffn_dim: inner dimension of the feed-forward block
        - window_size: sliding-window attention size for SWA

    Reuses from TransformerParameter:
        - seed, epoch, optimizer, initialization, loss (via NeuralNetworkParameter)
        - word_embedding_length → stored as L = word_embedding_length + 1
        - multi_head_attention_length → N (number of query heads)
        - vocabulary_length → V (number of output classes)
        - epsilon → for RMSNorm numerical stability
        - num_layers → number of stacked Mistral blocks

    Unused TransformerParameter fields (passed as empty/dummy):
        - input_hidden_layers, output_hidden_layers
        - input_activation_functions, output_activation_functions
        - gamma_input_values, gamma_output_values
        - beta_input_values, beta_output_values
        These are LayerNorm specific — Mistral uses RMSNorm instead.
    """

    __n_kv_heads: int   # number of key/value heads (GQA: n_kv_heads <= n_heads)
    __n_layers: int     # number of stacked Mistral blocks (not in TransformerParameter)
    __ffn_dim: int      # inner dimension of the feed-forward block
    __window_size: int  # sliding-window attention size (SWA)

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

        Maps Mistral hyperparameters to TransformerParameter fields:
            d_model       → word_embedding_length = d_model - 1
                            (TransformerParameter stores L = d_model internally)
            n_heads       → multi_head_attention_length
            vocab_size    → vocabulary_length
            n_layers      → num_layers
            epsilon       → epsilon

        :param seed: Random seed for reproducibility.
        :param epoch: Number of training epochs.
        :param optimizer: Optimization algorithm (e.g. AdamW).
        :param initialization: Weight initialization method.
        :param loss: Loss function (e.g. CrossEntropyLoss).
        :param d_model: Embedding and hidden dimension. Must be divisible by n_heads.
        :param n_heads: Number of query attention heads.
        :param n_kv_heads: Number of key/value heads for GQA.
                           Must satisfy n_heads % n_kv_heads == 0.
        :param n_layers: Number of stacked Mistral transformer blocks.
        :param ffn_dim: Inner dimension of the feed-forward network.
        :param window_size: Sliding-window attention size for SWA.
        :param vocab_size: Vocabulary / output class count.
        :param epsilon: Small constant for RMSNorm numerical stability.
        """
        # Validate Mistral-specific constraints before calling super()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})."
            )
        if (d_model // n_heads) % 2 != 0:
            raise ValueError(
                f"head_dim (d_model // n_heads = {d_model // n_heads}) "
                f"must be even for RoPE."
            )

        # Call TransformerParameter.__init__ with mapped arguments.
        # TransformerParameter stores L = word_embedding_length + 1 internally,
        # so we pass d_model - 1 to get L = d_model after the +1.
        # Unused LayerNorm fields (gamma, beta, hidden layers) passed as empty lists
        # since Mistral uses RMSNorm implemented directly in MistralModel.
        # TransformerParameter does not have num_layers — store it privately
        super().__init__(
            seed=seed,
            epoch=epoch,
            optimizer=optimizer,
            initialization=initialization,
            loss=loss,
            word_embedding_length=d_model - 1,   # TransformerParameter: L = this + 1 = d_model
            multi_head_attention_length=n_heads,  # N = number of query heads
            vocabulary_length=vocab_size,         # V = output class count
            epsilon=epsilon,                      # for RMSNorm
            input_hidden_layers=[],               # unused — Mistral uses SiLU FFN
            output_hidden_layers=[],              # unused
            input_activation_functions=[],        # unused
            output_activation_functions=[],       # unused
            gamma_input_values=[],                # unused — no LayerNorm
            gamma_output_values=[],               # unused
            beta_input_values=[],                 # unused
            beta_output_values=[]                 # unused
        )

        # Store Mistral-specific parameters not in TransformerParameter
        self.__n_kv_heads = n_kv_heads
        self.__n_layers = n_layers      # TransformerParameter has no num_layers field
        self.__ffn_dim = ffn_dim
        self.__window_size = window_size

    # ------------------------------------------------------------------
    # Mistral-specific getters
    # ------------------------------------------------------------------

    def getDModel(self) -> int:
        """
        Returns the embedding / hidden dimension.

        Maps to TransformerParameter.getL() which stores d_model.

        :return: d_model value.
        """
        # TransformerParameter stores L = word_embedding_length + 1 = d_model
        return self.getL()

    def getNHeads(self) -> int:
        """
        Returns the number of query attention heads.

        Maps to TransformerParameter.getN().

        :return: n_heads value.
        """
        return self.getN()

    def getNKVHeads(self) -> int:
        """
        Returns the number of key/value attention heads (GQA).

        :return: n_kv_heads value.
        """
        return self.__n_kv_heads

    def getNLayers(self) -> int:
        """
        Returns the number of stacked Mistral blocks.

        Stored privately since TransformerParameter has no num_layers field.

        :return: n_layers value.
        """
        return self.__n_layers

    def getFFNDim(self) -> int:
        """
        Returns the inner dimension of the feed-forward network.

        :return: ffn_dim value.
        """
        return self.__ffn_dim

    def getWindowSize(self) -> int:
        """
        Returns the sliding-window attention size.

        :return: window_size value.
        """
        return self.__window_size

    def getVocabSize(self) -> int:
        """
        Returns the vocabulary / output class size.

        Maps to TransformerParameter.getV().

        :return: vocab_size value.
        """
        return self.getV()

    def getHeadDim(self) -> int:
        """
        Returns the dimension of each query attention head.

        head_dim = d_model // n_heads
        Analogous to TransformerParameter.getDk().

        :return: Per-head dimension.
        """
        # getDk() in TransformerParameter = L // N = d_model // n_heads
        return self.getDk()

    def getKVHeadDim(self) -> int:
        """
        Returns the dimension of each key/value attention head.

        Same as head_dim since KV heads have the same per-head width.

        :return: Per KV-head dimension.
        """
        return self.getDk()

    def getGroupSize(self) -> int:
        """
        Returns how many query heads share each key/value head (GQA group size).

        group_size = n_heads // n_kv_heads

        :return: GQA group size.
        """
        return self.getN() // self.__n_kv_heads