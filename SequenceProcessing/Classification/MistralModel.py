from typing import List
import math
import random

from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode

from Math.Tensor import Tensor

from SequenceProcessing.Functions.Inverse import Inverse
from SequenceProcessing.Functions.SiLU import SiLU
from SequenceProcessing.Functions.SlidingWindowMask import SlidingWindowMask
from SequenceProcessing.Functions.SquareRoot import SquareRoot
from SequenceProcessing.Functions.Transpose import Transpose
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.Variance import Variance
from SequenceProcessing.Functions.RotaryPositionEmbedding import RotaryPositionEmbedding
from SequenceProcessing.Parameters.MistralParameter import MistralParameter


class MistralModel(ComputationalGraph):
    """
    Simplified Mistral-like transformer model.

    Implements the architecture shown in the Mistral diagram:
        Embedding
        → N × (RMSNorm → GQA with SWA → Residual
                → RMSNorm → FeedForward → Residual)
        → RMSNorm → Linear → Softmax → Output Probabilities

    Key Mistral features included:
        - RMSNorm instead of LayerNorm
        - Grouped Query Attention (GQA)
        - Sliding Window Attention (SWA) via SlidingWindowMask
        - SiLU activation in the feed-forward block
        - Residual connections around attention and FFN

    Input format (same as RecurrentNeuralNetworkModel):
        Each training instance is a flat 1D Tensor where the values
        are laid out as repeated (embedding_values..., class_label)
        groups, one group per time step.

    Usage:
        parameter = MistralParameter(...)
        model = MistralModel(parameter, word_embedding_length)
        model.train(train_set)
        accuracy = model.test(test_set)
    """

    __wordEmbeddingLength: int

    def __init__(self,
                 parameter: NeuralNetworkParameter,
                 word_embedding_length: int):
        """
        Constructor for MistralModel.

        :param parameter: MistralParameter object holding all hyperparameters.
        :param word_embedding_length: Length of each word embedding vector.
        """
        super().__init__(parameter)
        self.__wordEmbeddingLength = word_embedding_length

    # ------------------------------------------------------------------
    # Private graph-building helpers
    # ------------------------------------------------------------------

    def __rmsNorm(self,
                  input_node: ComputationalNode,
                  gamma_node: ComputationalNode) -> ComputationalNode:
        """
        Builds an RMSNorm sub-graph and returns its output node.

        RMSNorm(x) = x / RMS(x) * gamma
        where RMS(x) = sqrt( (1/n) * sum(x_i^2) )

        Since Variance computes sum(x_i^2)/n row-wise, the graph is:
            x → Variance → SquareRoot(epsilon) → Inverse
                         → hadamard(x, 1/RMS)
                         → hadamard(gamma)

        No mean subtraction — this is the key difference from LayerNorm.

        :param input_node: Node whose output will be normalised.
        :param gamma_node: Learnable scale parameter node (shape: 1 × d_model).
        :return: Output node after RMSNorm and gamma scaling.
        """
        parameter = self.parameters

        # Compute RMS(x)^2 = mean of squares, row-wise
        variance_node = self.addEdge(input_node, Variance())

        # sqrt(epsilon + RMS^2) → 1/RMS
        sqrt_node = self.addEdge(variance_node, SquareRoot(parameter.getEpsilon()))
        inv_node = self.addEdge(sqrt_node, Inverse())

        # x / RMS  (element-wise hadamard of input and 1/RMS)
        normalised = self.addEdge(input_node, inv_node, False, True)

        # Scale by learnable gamma
        return self.addEdge(normalised, gamma_node, False, True)

    def __groupedQueryAttention(self,
                                input_node: ComputationalNode,
                                random_generator: random.Random) -> ComputationalNode:
        """
        Builds a grouped query attention sub-graph.

        GQA groups query heads so that each group of (n_heads / n_kv_heads)
        query heads shares one key head and one value head.

        For each KV head k (0 .. n_kv_heads-1):
            - One W_K and W_V are initialised
            - group_size query heads each get their own W_Q
            - Each query head computes:
                  scores = (Q @ K^T) / sqrt(d_k)
                  scores = SlidingWindowMask(scores)
                  weights = Softmax(scores)
                  head_out = weights @ V
            - All head outputs are concatenated along axis=1

        :param input_node: Node providing the sequence representation
                           (shape: seq_len × d_model).
        :param random_generator: Random generator for weight initialisation.
        :return: Concatenated multi-head attention output node.
        """
        parameter = self.parameters
        d_model = parameter.getDModel()
        head_dim = parameter.getHeadDim()
        n_kv_heads = parameter.getNKVHeads()
        group_size = parameter.getGroupSize()
        window_size = parameter.getWindowSize()

        head_nodes = []

        for kv_idx in range(n_kv_heads):

            # Shared K and V weight matrices for this KV group
            w_k = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(d_model, head_dim, random_generator),
                    (d_model, head_dim)
                )
            )
            w_v = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(d_model, head_dim, random_generator),
                    (d_model, head_dim)
                )
            )

            # K and V projections (shared across all Q heads in this group)
            k = self.addEdge(input_node, w_k)
            v = self.addEdge(input_node, w_v)

            # Apply RoPE to K before transpose
            k = self.addEdge(k, RotaryPositionEmbedding(head_dim))
            k_transpose = self.addEdge(k, Transpose())

            for _ in range(group_size):

                # Each query head has its own W_Q
                w_q = MultiplicationNode(
                    Tensor(
                        parameter.initializeWeights(d_model, head_dim, random_generator),
                        (d_model, head_dim)
                    )
                )
                q = self.addEdge(input_node, w_q)

                # Apply RoPE to Q and K before the dot product
                # This encodes relative position into attention scores
                q = self.addEdge(q, RotaryPositionEmbedding(head_dim))

                # Scaled dot-product attention scores: Q @ K^T / sqrt(d_k)
                qk = self.addEdge(q, k_transpose, False, False)
                qk_scaled = self.addEdge(
                    qk,
                    MultiplyByConstant(1.0 / math.sqrt(head_dim))
                )

                # Sliding-window causal mask then softmax
                masked = self.addEdge(qk_scaled, SlidingWindowMask(window_size))
                weights = self.addEdge(masked, Softmax())

                # Weighted sum over values
                head_out = self.addEdge(weights, v)
                head_nodes.append(head_out)

        # Concatenate all head outputs along the feature axis
        return self.concatEdges(head_nodes, 1)

    def __feedForwardBlock(self,
                           input_node: ComputationalNode,
                           random_generator: random.Random) -> ComputationalNode:
        """
        Builds the feed-forward network sub-graph.

        Structure:
            x → Linear(d_model → ffn_dim) → SiLU → Linear(ffn_dim → d_model)

        SiLU is Mistral's activation function for the FFN block.
        The output dimension matches d_model so the residual connection
        can be added directly after.

        :param input_node: Input node (shape: seq_len × d_model).
        :param random_generator: Random generator for weight initialisation.
        :return: FFN output node (shape: seq_len × d_model).
        """
        parameter = self.parameters
        d_model = parameter.getDModel()
        ffn_dim = parameter.getFFNDim()

        # First linear: expand to ffn_dim
        w_up = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(d_model, ffn_dim, random_generator),
                (d_model, ffn_dim)
            )
        )
        hidden = self.addEdge(input_node, w_up)

        # SiLU activation
        activated = self.addEdge(hidden, SiLU(), True)

        # Second linear: project back to d_model
        w_down = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(ffn_dim + 1, d_model, random_generator),
                (ffn_dim + 1, d_model)
            )
        )
        return self.addEdge(activated, w_down)

    def __mistralBlock(self,
                       input_node: ComputationalNode,
                       random_generator: random.Random) -> ComputationalNode:
        """
        Builds one complete Mistral transformer block.

        Structure (matching the diagram):
            x
            ↓
            RMSNorm
            ↓
            GroupedQueryAttention (with SlidingWindowMask)
            ↓
            + x  (residual connection)
            ↓
            RMSNorm
            ↓
            FeedForward
            ↓
            + x  (residual connection)

        :param input_node: Input node to this block.
        :param random_generator: Random generator for weight initialisation.
        :return: Output node of this block.
        """
        parameter = self.parameters
        d_model = parameter.getDModel()

        # --- Attention sub-block ---

        # Gamma for first RMSNorm (initialised to ones → identity at start)
        # Must be MultiplicationNode with is_constant=True, matching Transformer.py
        gamma_attn_data = [1.0] * d_model
        gamma_attn = MultiplicationNode(
            True, False, Tensor(gamma_attn_data, (1, d_model)), True
        )

        normed_attn = self.__rmsNorm(input_node, gamma_attn)
        attn_out = self.__groupedQueryAttention(normed_attn, random_generator)

        # Residual: attention output + block input
        after_attn = self.addAdditionEdge(input_node, attn_out, False)

        # --- FFN sub-block ---

        # Gamma for second RMSNorm
        # Must be MultiplicationNode with is_constant=True, matching Transformer.py
        gamma_ffn_data = [1.0] * d_model
        gamma_ffn = MultiplicationNode(
            True, False, Tensor(gamma_ffn_data, (1, d_model)), True
        )

        normed_ffn = self.__rmsNorm(after_attn, gamma_ffn)
        ffn_out = self.__feedForwardBlock(normed_ffn, random_generator)

        # Residual: FFN output + post-attention node
        return self.addAdditionEdge(after_attn, ffn_out, False)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def __findTimeStep(self, train_set: List[Tensor]) -> int:
        """
        Returns the maximum sequence length across all instances.

        :param train_set: List of training tensors.
        :return: Maximum time step (sequence length).
        """
        time_step = -1
        for tensor in train_set:
            size = tensor.getShape()[0]
            current = size // (self.__wordEmbeddingLength + 1)
            if time_step < current:
                time_step = current
        return time_step

    def __createInputTensors(self, instance: Tensor) -> List[int]:
        """
        Prepares the input node value from a flat sequence tensor and
        returns the ground-truth class labels.

        Input format: flat 1D tensor with layout
            [emb_0, emb_1, ..., emb_{L-1}, label,
             emb_0, emb_1, ..., emb_{L-1}, label, ...]
        one group per time step.

        Sets self.input_nodes[0] to a 2D tensor of shape
        (time_step, word_embedding_length).

        :param instance: Flat input tensor.
        :return: List of integer class labels, one per time step.
        """
        class_labels = []
        embedding_values = []

        time_step = instance.getShape()[0] // (self.__wordEmbeddingLength + 1)
        j = 0

        for _ in range(time_step):
            for _ in range(self.__wordEmbeddingLength):
                embedding_values.append(instance.getValue((j,)))
                j += 1
            class_labels.append(int(instance.getValue((j,))))
            j += 1

        self.input_nodes[0].setValue(
            Tensor(embedding_values, (time_step, self.__wordEmbeddingLength))
        )

        return class_labels

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, train_set: List[Tensor]) -> None:
        """
        Builds the Mistral graph and runs training.

        Graph structure:
            input_node (seq_len × d_model — set per instance)
            ↓
            N × mistralBlock
            ↓
            RMSNorm (final)
            ↓
            Linear (d_model → vocab_size)
            ↓
            Softmax
            ↓
            output_node

        :param train_set: List of flat sequence tensors.
        """
        parameter = self.parameters
        random_generator = random.Random(parameter.getSeed())

        d_model = parameter.getDModel()
        vocab_size = parameter.getVocabSize()

        # Single input node -- receives the raw sequence as
        # (seq_len x word_embedding_length)
        input_node = MultiplicationNode(False, True)
        self.input_nodes.append(input_node)

        # Embedding projection: word_embedding_length -> d_model
        # This is the 'Embedding' box in the Mistral diagram.
        # Every subsequent layer expects (seq_len x d_model).
        # The input node has is_biased=True so the framework appends a bias
        # column, making the actual input shape (seq_len x word_embedding_length+1).
        # w_embed must have word_embedding_length+1 rows to match.
        w_embed = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(
                    self.__wordEmbeddingLength + 1, d_model, random_generator
                ),
                (self.__wordEmbeddingLength + 1, d_model)
            )
        )
        embedded = self.addEdge(input_node, w_embed)

        # Stack N Mistral blocks -- all operating on (seq_len x d_model)
        current = embedded
        for _ in range(parameter.getNLayers()):
            current = self.__mistralBlock(current, random_generator)

        # Final RMSNorm before the output head
        # Must be MultiplicationNode with is_constant=True, matching Transformer.py
        gamma_final_data = [1.0] * d_model
        gamma_final = MultiplicationNode(
            True, False, Tensor(gamma_final_data, (1, d_model)), True
        )
        current = self.__rmsNorm(current, gamma_final)

        # Output head: Linear → Softmax
        w_out = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(d_model, vocab_size, random_generator),
                (d_model, vocab_size)
            )
        )
        logits = self.addEdge(current, w_out)
        self.output_node = self.addEdge(logits, Softmax())

        # Class label node (ground truth target)
        class_label_node = ComputationalNode()
        self.input_nodes.append(class_label_node)

        loss_inputs = [self.output_node, class_label_node]
        self.addFunctionEdge(loss_inputs, parameter.getLossFunction(), False)

        # Training loop
        for _ in range(parameter.getEpoch()):

            # Shuffle training set
            for _ in range(len(train_set)):
                i1 = random_generator.randint(0, len(train_set) - 1)
                i2 = random_generator.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            for instance in train_set:
                class_labels = self.__createInputTensors(instance)

                # Build one-hot class label tensor (seq_len × vocab_size)
                class_label_values = []
                for class_label in class_labels:
                    for j in range(vocab_size):
                        class_label_values.append(1.0 if j == class_label else 0.0)

                self.input_nodes[1].setValue(
                    Tensor(class_label_values, (len(class_labels), vocab_size))
                )

                self.forwardCalculation()
                self.backpropagation()

            parameter.getOptimizer().setLearningRate()

    def getOutputValue(self, output_node: ComputationalNode) -> List[float]:
        """
        Extracts predicted class indices from the output node.

        For each row (time step) takes the argmax across vocab_size columns.

        :param output_node: The model output node after Softmax.
        :return: List of predicted class indices as floats.
        """
        class_labels = []
        shape = output_node.getValue().getShape()

        for i in range(shape[0]):
            max_val = float("-inf")
            index = -1

            for j in range(shape[1]):
                val = output_node.getValue().getValue((i, j))
                if val > max_val:
                    max_val = val
                    index = j

            class_labels.append(float(index))

        return class_labels

    def test(self, test_set: List[Tensor]) -> float:
        """
        Evaluates the model on a test set and returns accuracy.

        :param test_set: List of flat sequence tensors.
        :return: Accuracy as a float in [0, 1].
        """
        count = 0
        total = 0

        for instance in test_set:
            gold = self.__createInputTensors(instance)
            pred = self.predict()

            time_step = instance.getShape()[0] // (self.__wordEmbeddingLength + 1)

            for j in range(time_step):
                if gold[j] == int(pred[j]):
                    count += 1
                total += 1

        return count / total