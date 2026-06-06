from typing import List
import math
import random

from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode

from Math.Tensor import Tensor

from SequenceProcessing.Classification.Transformer import Transformer
from SequenceProcessing.Functions.Inverse import Inverse
from SequenceProcessing.Functions.SiLU import SiLU
from SequenceProcessing.Functions.SlidingWindowMask import SlidingWindowMask
from SequenceProcessing.Functions.SquareRoot import SquareRoot
from SequenceProcessing.Functions.Transpose import Transpose
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.Variance import Variance
from SequenceProcessing.Functions.RotaryPositionEmbedding import RotaryPositionEmbedding
from SequenceProcessing.Parameters.MistralParameter import MistralParameter


class MistralModel(Transformer):
    """
    Simplified Mistral-like decoder-only transformer model.

    Extends Transformer and overrides train(), test(), getOutputValue(),
    and createInputTensors() to implement Mistral-specific architecture:

        Embedding projection
        → N x (RMSNorm → GQA with RoPE and SWA → Residual
                → RMSNorm → SiLU FeedForward → Residual)
        → RMSNorm → Linear → Softmax → Output Probabilities

    Mistral-specific features vs base Transformer:
        - RMSNorm instead of LayerNorm (no mean subtraction)
        - Grouped Query Attention (GQA): multiple Q heads share one K/V
        - Sliding Window Attention (SWA): limits attention lookback
        - Rotary Position Embedding (RoPE): rotates Q and K by position
        - SiLU activation in feed-forward block
        - Decoder-only (no encoder stack)

    Input format:
        Flat 1D Tensor: [emb_0...emb_{L-1}, label, emb_0...emb_{L-1}, label, ...]
        One group per time step.
    """

    __wordEmbeddingLength: int

    def __init__(self,
                 parameter: MistralParameter,
                 word_embedding_length: int):
        """
        Constructor for MistralModel.

        :param parameter: MistralParameter holding all hyperparameters.
        :param word_embedding_length: Length of each word embedding vector.
        """
        # Pass a dummy VectorizedDictionary to satisfy Transformer.__init__
        # MistralModel does not use the dictionary — it takes raw embeddings
        from Dictionary.VectorizedDictionary import VectorizedDictionary
        dummy_dict = VectorizedDictionary()
        super().__init__(parameter, dummy_dict)
        self.__wordEmbeddingLength = word_embedding_length

    # ------------------------------------------------------------------
    # Private graph-building helpers
    # ------------------------------------------------------------------

    def __rmsNorm(self,
                  input_node: ComputationalNode,
                  gamma_node: ComputationalNode) -> ComputationalNode:
        """
        Builds an RMSNorm sub-graph.

        RMSNorm(x) = (x / RMS(x)) * gamma
        where RMS(x) = sqrt( mean(x^2) ) = sqrt( sum(x_i^2) / n )

        Unlike LayerNorm there is no mean subtraction — only RMS scaling.
        This is the key architectural difference from the base Transformer.

        Graph chain:
            x → Variance → SquareRoot(epsilon) → Inverse
              → hadamard(x, 1/RMS) → hadamard(gamma)

        :param input_node: Node whose output will be normalised.
        :param gamma_node: Learnable scale parameter (shape: 1 x d_model).
        :return: Output node after RMSNorm and gamma scaling.
        """
        parameter = self.parameters

        # Step 1: Variance computes mean(x^2) = sum(x_i^2) / n  →  RMS(x)^2
        variance_node = self.addEdge(input_node, Variance())

        # Step 2: sqrt(epsilon + RMS^2) → RMS(x)
        # epsilon added for numerical stability (avoid sqrt(0))
        sqrt_node = self.addEdge(variance_node, SquareRoot(parameter.getEpsilon()))

        # Step 3: 1 / RMS(x)
        inv_node = self.addEdge(sqrt_node, Inverse())

        # Step 4: x / RMS(x)  — element-wise: input * (1/RMS)
        normalised = self.addEdge(input_node, inv_node, False, True)

        # Step 5: scale by learnable gamma parameter
        # gamma initialised to 1.0 so initially this is identity
        return self.addEdge(normalised, gamma_node, False, True)

    def __groupedQueryAttention(self,
                                input_node: ComputationalNode,
                                random_generator: random.Random) -> ComputationalNode:
        """
        Builds a Grouped Query Attention (GQA) sub-graph with RoPE and SWA.

        GQA reduces K/V heads while keeping full Q heads:
            group_size = n_heads // n_kv_heads
            Each KV head serves group_size Q heads.

        For each KV head k:
            K = input @ W_K        shape: (seq, head_dim)
            V = input @ W_V        shape: (seq, head_dim)
            K = RoPE(K)            rotate by position
            K^T                    shape: (head_dim, seq)

            For each Q head in group:
                Q = input @ W_Q    shape: (seq, head_dim)
                Q = RoPE(Q)        rotate by position

                # Scaled dot-product attention:
                S = Q @ K^T / sqrt(d_k)    shape: (seq, seq)
                S = SlidingWindowMask(S)   mask future and old positions
                A = softmax(S)             attention weights sum to 1
                out = A @ V                shape: (seq, head_dim)

        All head outputs concatenated → shape: (seq, n_heads * head_dim)
                                              = (seq, d_model)

        :param input_node: Node providing sequence (seq_len x d_model).
        :param random_generator: For weight initialisation.
        :return: Concatenated multi-head attention output node.
        """
        parameter = self.parameters
        d_model = parameter.getDModel()
        head_dim = parameter.getHeadDim()
        n_kv_heads = parameter.getNKVHeads()
        group_size = parameter.getGroupSize()
        window_size = parameter.getWindowSize()

        head_nodes = []

        # Outer loop: one K and V per KV head (shared across group)
        for kv_idx in range(n_kv_heads):

            # W_K: (d_model, head_dim) — projects input to key space
            w_k = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(d_model, head_dim, random_generator),
                    (d_model, head_dim)
                )
            )
            # W_V: (d_model, head_dim) — projects input to value space
            w_v = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(d_model, head_dim, random_generator),
                    (d_model, head_dim)
                )
            )

            # K = input @ W_K    shape: (seq_len, head_dim)
            k = self.addEdge(input_node, w_k)

            # V = input @ W_V    shape: (seq_len, head_dim)
            v = self.addEdge(input_node, w_v)

            # Apply RoPE to K: rotate each dimension pair by position angle
            # K_rot[pos, i]   = K[pos,i]*cos(pos*theta_i) - K[pos,i+1]*sin(pos*theta_i)
            # K_rot[pos, i+1] = K[pos,i]*sin(pos*theta_i) + K[pos,i+1]*cos(pos*theta_i)
            k = self.addEdge(k, RotaryPositionEmbedding(head_dim))

            # K^T shape: (head_dim, seq_len) — for dot product with Q
            k_transpose = self.addEdge(k, Transpose())

            # Inner loop: group_size Q heads sharing the same K and V
            for _ in range(group_size):

                # W_Q: (d_model, head_dim) — each Q head has its own weights
                w_q = MultiplicationNode(
                    Tensor(
                        parameter.initializeWeights(d_model, head_dim, random_generator),
                        (d_model, head_dim)
                    )
                )

                # Q = input @ W_Q    shape: (seq_len, head_dim)
                q = self.addEdge(input_node, w_q)

                # Apply RoPE to Q: same rotation formula as K
                q = self.addEdge(q, RotaryPositionEmbedding(head_dim))

                # Scaled dot-product attention scores:
                # S = Q @ K^T    shape: (seq_len, seq_len)
                # S[i,j] = how much token i attends to token j
                qk = self.addEdge(q, k_transpose, False, False)

                # Scale: S = S / sqrt(d_k)
                # Prevents softmax saturation for large head_dim values
                qk_scaled = self.addEdge(
                    qk,
                    MultiplyByConstant(1.0 / math.sqrt(head_dim))
                )

                # Apply sliding window causal mask:
                # S[i,j] = -inf if j > i (future) or j < i-W (too old)
                # After softmax: -inf → 0 (those positions ignored)
                masked = self.addEdge(qk_scaled, SlidingWindowMask(window_size))

                # Softmax: convert scores to attention weights
                # A[i,:] sums to 1.0 — probability distribution over positions
                weights = self.addEdge(masked, Softmax())

                # Weighted sum of values:
                # out = A @ V    shape: (seq_len, head_dim)
                head_out = self.addEdge(weights, v)
                head_nodes.append(head_out)

        # Concatenate all n_heads outputs along feature axis:
        # (seq, head_dim) x n_heads → (seq, n_heads * head_dim) = (seq, d_model)
        return self.concatEdges(head_nodes, 1)

    def __feedForwardBlock(self,
                           input_node: ComputationalNode,
                           random_generator: random.Random) -> ComputationalNode:
        """
        Builds the SiLU feed-forward network sub-graph.

        Structure:
            x → Linear(d_model → ffn_dim) → SiLU → Linear(ffn_dim → d_model)

        Mathematically:
            hidden = x @ W_up                  shape: (seq, ffn_dim)
            activated = SiLU(hidden)           shape: (seq, ffn_dim+1) with bias
            output = activated @ W_down        shape: (seq, d_model)

        The expand-then-contract pattern allows the model to learn richer
        non-linear transformations in the larger ffn_dim space.

        :param input_node: Input node (seq_len x d_model).
        :param random_generator: For weight initialisation.
        :return: FFN output node (seq_len x d_model).
        """
        parameter = self.parameters
        d_model = parameter.getDModel()
        ffn_dim = parameter.getFFNDim()

        # W_up: (d_model, ffn_dim) — expand to larger space
        w_up = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(d_model, ffn_dim, random_generator),
                (d_model, ffn_dim)
            )
        )
        # hidden = x @ W_up    shape: (seq_len, ffn_dim)
        hidden = self.addEdge(input_node, w_up)

        # SiLU(x) = x * sigmoid(x) — Mistral's non-linear activation
        # is_biased=True: framework appends bias column → shape: (seq, ffn_dim+1)
        activated = self.addEdge(hidden, SiLU(), True)

        # W_down: (ffn_dim+1, d_model) — contract back to d_model
        # +1 accounts for bias column added by is_biased=True above
        w_down = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(ffn_dim + 1, d_model, random_generator),
                (ffn_dim + 1, d_model)
            )
        )
        # output = activated @ W_down    shape: (seq_len, d_model)
        return self.addEdge(activated, w_down)

    def __mistralBlock(self,
                       input_node: ComputationalNode,
                       random_generator: random.Random) -> ComputationalNode:
        """
        Builds one complete Mistral decoder block.

        Structure (from diagram):
            x
            ↓ RMSNorm
            ↓ GQA with RoPE + SWA
            ↓ + x  (residual: output = attention + input)
            ↓ RMSNorm
            ↓ FeedForward (SiLU)
            ↓ + x  (residual: output = ffn + post-attention)

        Residual connections ensure gradients flow without vanishing
        and preserve the original signal through deep networks.

        :param input_node: Input node to this block.
        :param random_generator: For weight initialisation.
        :return: Output node of this block.
        """
        parameter = self.parameters
        d_model = parameter.getDModel()

        # --- Attention sub-block ---

        # Gamma for first RMSNorm — initialised to 1.0 (identity at start)
        # Learnable: model adjusts scale after normalisation during training
        gamma_attn = MultiplicationNode(
            True, False, Tensor([1.0] * d_model, (1, d_model)), True
        )

        # RMSNorm: x / RMS(x) * gamma    shape unchanged: (seq, d_model)
        normed_attn = self.__rmsNorm(input_node, gamma_attn)

        # GQA attention with RoPE and SWA    shape: (seq, d_model)
        attn_out = self.__groupedQueryAttention(normed_attn, random_generator)

        # First residual connection: x + attention_output
        # Preserves original signal; allows gradient flow in deep networks
        after_attn = self.addAdditionEdge(input_node, attn_out, False)

        # --- FFN sub-block ---

        # Gamma for second RMSNorm — separate learnable scale
        gamma_ffn = MultiplicationNode(
            True, False, Tensor([1.0] * d_model, (1, d_model)), True
        )

        # RMSNorm before FFN    shape unchanged: (seq, d_model)
        normed_ffn = self.__rmsNorm(after_attn, gamma_ffn)

        # Feed-forward: expand → SiLU → contract    shape: (seq, d_model)
        ffn_out = self.__feedForwardBlock(normed_ffn, random_generator)

        # Second residual connection: post_attention + ffn_output
        return self.addAdditionEdge(after_attn, ffn_out, False)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def __createInputTensors(self, instance: Tensor) -> List[int]:
        """
        Prepares input node value from a flat sequence tensor.

        Input format (flat 1D tensor):
            [emb_0, ..., emb_{L-1}, label,
             emb_0, ..., emb_{L-1}, label, ...]  — one group per time step

        Sets input_nodes[0] to shape (time_step, word_embedding_length).
        Returns ground truth class labels as integer list.

        :param instance: Flat input tensor.
        :return: List of integer class labels per time step.
        """
        class_labels = []
        embedding_values = []

        # Each time step contributes (word_embedding_length + 1) values
        time_step = instance.getShape()[0] // (self.__wordEmbeddingLength + 1)
        j = 0

        for _ in range(time_step):
            # Collect word_embedding_length embedding values
            for _ in range(self.__wordEmbeddingLength):
                embedding_values.append(instance.getValue((j,)))
                j += 1
            # Collect class label (last value in group)
            class_labels.append(int(instance.getValue((j,))))
            j += 1

        # Set input node to 2D embedding matrix: (time_step, word_embedding_length)
        self.input_nodes[0].setValue(
            Tensor(embedding_values, (time_step, self.__wordEmbeddingLength))
        )

        return class_labels

    # ------------------------------------------------------------------
    # Public interface (overrides Transformer methods)
    # ------------------------------------------------------------------

    def train(self, train_set: List[Tensor]) -> None:
        """
        Builds the Mistral graph and runs training.

        Graph structure:
            input (seq x word_embedding_length)
            ↓ w_embed: project word_embedding_length → d_model
            ↓ N x mistralBlock
            ↓ final RMSNorm
            ↓ w_out: project d_model → vocab_size
            ↓ Softmax → output probabilities

        :param train_set: List of flat sequence tensors.
        """
        parameter = self.parameters
        random_generator = random.Random(parameter.getSeed())

        d_model = parameter.getDModel()
        vocab_size = parameter.getVocabSize()

        # Input node receives raw word embeddings: (seq_len, word_embedding_length)
        # is_biased=True: framework appends bias column automatically
        input_node = MultiplicationNode(False, True)
        self.input_nodes.append(input_node)

        # Embedding projection: word_embedding_length → d_model
        # W_embed shape: (word_embedding_length+1, d_model)
        # +1 accounts for bias column appended by is_biased=True on input_node
        # After projection: (seq_len, d_model) — all subsequent layers use d_model
        w_embed = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(
                    self.__wordEmbeddingLength + 1, d_model, random_generator
                ),
                (self.__wordEmbeddingLength + 1, d_model)
            )
        )
        # embedded = input @ W_embed    shape: (seq_len, d_model)
        embedded = self.addEdge(input_node, w_embed)

        # Stack N Mistral decoder blocks
        # Each block: RMSNorm → GQA+RoPE+SWA → residual → RMSNorm → FFN → residual
        current = embedded
        for _ in range(parameter.getNLayers()):
            current = self.__mistralBlock(current, random_generator)

        # Final RMSNorm after all blocks
        # RMSNorm(x) = x / RMS(x) * gamma
        gamma_final = MultiplicationNode(
            True, False, Tensor([1.0] * d_model, (1, d_model)), True
        )
        current = self.__rmsNorm(current, gamma_final)

        # Output head: project d_model → vocab_size
        # W_out shape: (d_model, vocab_size)
        w_out = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(d_model, vocab_size, random_generator),
                (d_model, vocab_size)
            )
        )
        # logits = current @ W_out    shape: (seq_len, vocab_size)
        logits = self.addEdge(current, w_out)

        # Softmax: convert logits to probabilities
        # output[i,j] = probability that token i has label j
        self.output_node = self.addEdge(logits, Softmax())

        # Class label node — receives one-hot ground truth during training
        class_label_node = ComputationalNode()
        self.input_nodes.append(class_label_node)

        # CrossEntropyLoss: loss = -sum(y_true * log(y_pred))
        loss_inputs = [self.output_node, class_label_node]
        self.addFunctionEdge(loss_inputs, parameter.getLossFunction(), False)

        # Training loop
        for _ in range(parameter.getEpoch()):

            # Shuffle training set each epoch
            for _ in range(len(train_set)):
                i1 = random_generator.randint(0, len(train_set) - 1)
                i2 = random_generator.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            for instance in train_set:
                # Set input embeddings and get ground truth labels
                class_labels = self.__createInputTensors(instance)

                # Build one-hot label tensor: shape (time_step, vocab_size)
                # one_hot[t, class_labels[t]] = 1.0, rest = 0.0
                class_label_values = []
                for class_label in class_labels:
                    for j in range(vocab_size):
                        class_label_values.append(1.0 if j == class_label else 0.0)

                self.input_nodes[1].setValue(
                    Tensor(class_label_values, (len(class_labels), vocab_size))
                )

                # Forward pass: compute predictions and loss
                self.forwardCalculation()

                # Backward pass: compute gradients via chain rule
                self.backpropagation()

            # Update all weights using AdamW optimizer
            parameter.getOptimizer().setLearningRate()

    def getOutputValue(self, output_node: ComputationalNode) -> List[float]:
        """
        Extracts predicted class indices from the output node.

        For each time step i: predicted_label = argmax over vocab_size columns.

        :param output_node: Model output node after Softmax.
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

            # argmax: index of highest probability = predicted label
            class_labels.append(float(index))

        return class_labels

    def test(self, test_set: List[Tensor]) -> float:
        """
        Evaluates the model on a test set and returns word-level accuracy.

        accuracy = correct_predictions / total_words

        :param test_set: List of flat sequence tensors.
        :return: Accuracy as a float in [0, 1].
        """
        count = 0
        total = 0

        for instance in test_set:
            # Set input and get ground truth labels
            gold = self.__createInputTensors(instance)

            # Get model predictions via forward pass
            pred = self.predict()

            time_step = instance.getShape()[0] // (self.__wordEmbeddingLength + 1)

            # Compare predicted label vs gold label for each time step
            for j in range(time_step):
                if gold[j] == int(pred[j]):
                    count += 1
                total += 1

        # Word-level accuracy = correct / total
        return count / total