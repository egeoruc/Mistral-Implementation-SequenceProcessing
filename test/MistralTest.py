import math
import unittest

from Math.Tensor import Tensor

from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.AdamW import AdamW

from SequenceProcessing.Classification.MistralModel import MistralModel
from SequenceProcessing.Functions.SiLU import SiLU
from SequenceProcessing.Functions.SlidingWindowMask import SlidingWindowMask
from SequenceProcessing.Functions.RotaryPositionEmbedding import RotaryPositionEmbedding
from SequenceProcessing.Parameters.MistralParameter import MistralParameter
from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus
from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord


class MistralTest(unittest.TestCase):
    """
    Unit tests for the simplified Mistral-like model.

    Follows the same pattern as TransformerTest:
        - Small synthetic tensors for fast graph/training tests
        - Real corpus test on postag-atis-en dataset
        - Function tests for SiLU, SlidingWindowMask, RotaryPositionEmbedding
    """

    # ------------------------------------------------------------------
    # Helper — shared parameter factory
    # ------------------------------------------------------------------

    def __makeParameter(self, seed: int = 42,
                        epoch: int = 1,
                        d_model: int = 8,
                        n_heads: int = 2,
                        n_kv_heads: int = 1,
                        n_layers: int = 1,
                        ffn_dim: int = 8,
                        window_size: int = 2,
                        vocab_size: int = 4) -> MistralParameter:
        """
        Creates a small MistralParameter for testing.

        :return: MistralParameter instance.
        """
        return MistralParameter(
            seed=seed,
            epoch=epoch,
            optimizer=AdamW(0.01, 0.9, 0.9, 0.999, 1e-8, 0.01),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            window_size=window_size,
            vocab_size=vocab_size,
            epsilon=1e-6
        )

    # ------------------------------------------------------------------
    # Model tests
    # ------------------------------------------------------------------

    def testInitialization(self):
        """
        Tests that the model graph builds and trains without errors.

        Verifies:
            - MistralParameter validates correctly
            - MistralModel constructs the computational graph
            - Forward pass runs without shape mismatches
            - Backpropagation runs without errors
            - Optimizer step runs
            - Training completes for all epochs

        Input format (word_embedding_length=3, vocab_size=4):
            Each time step: [emb_0, emb_1, emb_2, class_label]
            Two time steps = 8 values, shape (8,).
        """
        tensors = [
            Tensor([0.2, 0.5, 0.1, 2, 0.4, 0.3, 0.7, 1], (8,)),
            Tensor([0.6, 0.1, 0.9, 0, 0.3, 0.8, 0.2, 3], (8,)),
            Tensor([0.5, 0.5, 0.5, 1, 0.1, 0.9, 0.4, 2], (8,)),
        ]

        # d_model=8, n_heads=2 → head_dim=4 (even, valid for RoPE)
        # n_heads=2, n_kv_heads=1 → group_size=2 (GQA)
        parameter = self.__makeParameter(epoch=2)
        model = MistralModel(parameter, word_embedding_length=3)
        model.train(tensors)

    def testParameterValidation(self):
        """
        Tests that MistralParameter raises ValueError on invalid configs.

        Checks three validation rules:
            1. d_model must be divisible by n_heads
            2. n_heads must be divisible by n_kv_heads
            3. head_dim (d_model // n_heads) must be even for RoPE
        """
        base_args = dict(
            seed=1, epoch=1,
            optimizer=AdamW(0.01, 0.9, 0.9, 0.999, 1e-8, 0.01),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            n_layers=1, ffn_dim=8, window_size=2, vocab_size=4, epsilon=1e-6,
        )

        # Rule 1: d_model=7 not divisible by n_heads=2
        with self.assertRaises(ValueError):
            MistralParameter(d_model=7, n_heads=2, n_kv_heads=1, **base_args)

        # Rule 2: n_heads=4 not divisible by n_kv_heads=3
        with self.assertRaises(ValueError):
            MistralParameter(d_model=8, n_heads=4, n_kv_heads=3, **base_args)

        # Rule 3: head_dim=3 (odd) — d_model=6, n_heads=2 → head_dim=3
        with self.assertRaises(ValueError):
            MistralParameter(d_model=6, n_heads=2, n_kv_heads=1, **base_args)

    def testGetOutputValue(self):
        """
        Tests that test() returns a valid accuracy in [0, 1].
        """
        parameter = self.__makeParameter(seed=7)
        tensors = [
            Tensor([0.2, 0.5, 0.1, 2, 0.4, 0.3, 0.7, 1], (8,)),
            Tensor([0.6, 0.1, 0.9, 0, 0.3, 0.8, 0.2, 3], (8,)),
        ]
        model = MistralModel(parameter, word_embedding_length=3)
        model.train(tensors)

        accuracy = model.test(tensors)

        # Accuracy must be a valid probability in [0, 1]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def testMultipleLayersRun(self):
        """
        Tests that stacking n_layers=2 blocks builds and trains without errors.
        Also tests n_kv_heads=n_heads which degrades to standard MHA.
        """
        parameter = self.__makeParameter(
            seed=3, n_heads=2, n_kv_heads=2, n_layers=2, window_size=3
        )
        tensors = [
            Tensor([0.2, 0.5, 0.1, 2, 0.4, 0.3, 0.7, 1], (8,)),
            Tensor([0.6, 0.1, 0.9, 0, 0.3, 0.8, 0.2, 3], (8,)),
        ]
        model = MistralModel(parameter, word_embedding_length=3)
        model.train(tensors)

    def testOnRealDataset(self):
        """
        Tests MistralModel on a small subset of the postag-atis-en dataset.

        Verifies end-to-end pipeline:
            SequenceCorpus → tensors → train → test → valid accuracy

        Uses 30 training and 10 test sentences to keep test fast.
        """
        train_corpus = SequenceCorpus("Datasets/postag-atis-en-train.txt")
        test_corpus = SequenceCorpus("Datasets/postag-atis-en-test.txt")

        # Build label vocabulary from training corpus
        class_labels = train_corpus.getClassLabels()
        label_to_index = {label: i for i, label in enumerate(class_labels)}
        vocab_size = len(class_labels)

        def to_tensors(corpus, limit):
            """Convert SequenceCorpus to flat tensors."""
            tensors = []
            word_embedding_length = 8
            for i in range(min(limit, corpus.sentenceCount())):
                sentence = corpus.getSentence(i)
                values = []
                for j in range(sentence.wordCount()):
                    word = sentence.getWord(j)
                    if isinstance(word, LabelledVectorizedWord):
                        # Zero embeddings — no pre-trained vectors available
                        for _ in range(word_embedding_length):
                            values.append(0.0)
                        values.append(float(
                            label_to_index.get(word.getClassLabel(), 0)
                        ))
                if values:
                    tensors.append(Tensor(values, (len(values),)))
            return tensors

        train_tensors = to_tensors(train_corpus, limit=30)
        test_tensors = to_tensors(test_corpus, limit=10)

        parameter = MistralParameter(
            seed=1, epoch=1,
            optimizer=AdamW(0.01, 0.9, 0.9, 0.999, 1e-8, 0.01),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            d_model=8, n_heads=2, n_kv_heads=1, n_layers=1,
            ffn_dim=16, window_size=4, vocab_size=vocab_size, epsilon=1e-6
        )

        model = MistralModel(parameter, word_embedding_length=8)
        model.train(train_tensors)
        accuracy = model.test(test_tensors)

        # Accuracy must be a valid value in [0, 1]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    # ------------------------------------------------------------------
    # SiLU Function tests
    # ------------------------------------------------------------------

    def testSiLUCalculateAtZero(self):
        """
        SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0.
        """
        tensor = Tensor([0.0, 0.0], (1, 2))
        result = SiLU().calculate(tensor)
        for val in result.getData():
            self.assertAlmostEqual(val, 0.0, places=6)

    def testSiLUCalculatePositive(self):
        """
        SiLU(1) = 1 * sigmoid(1) = sigmoid(1) ≈ 0.7310585.
        """
        tensor = Tensor([1.0], (1, 1))
        result = SiLU().calculate(tensor)
        expected = 1.0 / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(result.getData()[0], expected, places=6)

    def testSiLUCalculateNegative(self):
        """
        SiLU of negative values should be negative.
        """
        tensor = Tensor([-1.0, -2.0], (1, 2))
        result = SiLU().calculate(tensor)
        for val in result.getData():
            self.assertLess(val, 0.0)

    def testSiLUShapePreserved(self):
        """
        SiLU output shape must match input shape.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        result = SiLU().calculate(tensor)
        self.assertEqual(result.getShape(), (2, 2))

    def testSiLUDerivativeAtZero(self):
        """
        SiLU'(0) = sigmoid(0) * (1 + 0) = 0.5.
        With backward = ones, result should be 0.5.
        """
        tensor = Tensor([0.0, 0.0], (1, 2))
        backward = Tensor([1.0, 1.0], (1, 2))
        result = SiLU().derivative(tensor, backward)
        for val in result.getData():
            self.assertAlmostEqual(val, 0.5, places=6)

    # ------------------------------------------------------------------
    # SlidingWindowMask Function tests
    # ------------------------------------------------------------------

    def testSlidingWindowMaskFutureMasked(self):
        """
        Future positions (j > i) must always be -inf regardless of window size.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        result = SlidingWindowMask(window_size=100).calculate(tensor)
        # (0,1) is future for token 0
        self.assertTrue(math.isinf(result.getData()[1]) and result.getData()[1] < 0)

    def testSlidingWindowMaskOldTokensMasked(self):
        """
        Positions older than window_size must be -inf.
        For window_size=1 and seq=3, position (2,0) should be masked.
        """
        values = [float(k) for k in range(1, 10)]
        tensor = Tensor(values, (3, 3))
        result = SlidingWindowMask(window_size=1).calculate(tensor)
        # Row 2 (i=2), col 0 (j=0): j < i - W → 0 < 2 - 1 = 1 → masked
        self.assertTrue(math.isinf(result.getData()[6]) and result.getData()[6] < 0)

    def testSlidingWindowMaskShapePreserved(self):
        """
        Output shape must match input shape.
        """
        tensor = Tensor([float(k) for k in range(9)], (3, 3))
        result = SlidingWindowMask(window_size=2).calculate(tensor)
        self.assertEqual(result.getShape(), (3, 3))

    def testSlidingWindowMaskLargeWindowIsCausal(self):
        """
        When window_size >= seq_len the mask degrades to standard causal mask.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        result = SlidingWindowMask(window_size=100).calculate(tensor)
        data = result.getData()
        # (0,0) keep, (0,1) future=-inf, (1,0) keep, (1,1) keep
        self.assertEqual(data[0], 1.0)
        self.assertTrue(math.isinf(data[1]) and data[1] < 0)
        self.assertEqual(data[2], 3.0)
        self.assertEqual(data[3], 4.0)

    def testSlidingWindowMaskInvalidWindowSize(self):
        """
        window_size < 1 must raise ValueError.
        """
        with self.assertRaises(ValueError):
            SlidingWindowMask(window_size=0)

    # ------------------------------------------------------------------
    # RotaryPositionEmbedding Function tests
    # ------------------------------------------------------------------

    def testRoPEShapePreserved(self):
        """
        RoPE output shape must match input shape.
        """
        tensor = Tensor([float(k) for k in range(8)], (2, 4))
        result = RotaryPositionEmbedding(head_dim=4).calculate(tensor)
        self.assertEqual(result.getShape(), (2, 4))

    def testRoPEPositionZeroUnchanged(self):
        """
        At position 0, cos(0)=1 and sin(0)=0 so the vector is unchanged.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 4))
        result = RotaryPositionEmbedding(head_dim=4).calculate(tensor)
        data = result.getData()
        # Position 0 row should be unchanged
        self.assertAlmostEqual(data[0], 1.0, places=6)
        self.assertAlmostEqual(data[1], 2.0, places=6)
        self.assertAlmostEqual(data[2], 3.0, places=6)
        self.assertAlmostEqual(data[3], 4.0, places=6)

    def testRoPEOddHeadDimRaisesError(self):
        """
        head_dim must be even — RoPE processes dimensions in pairs.
        """
        with self.assertRaises(ValueError):
            RotaryPositionEmbedding(head_dim=3)

    def testRoPEDerivativeShapePreserved(self):
        """
        Derivative output shape must match input shape.
        """
        tensor = Tensor([float(k) for k in range(8)], (2, 4))
        backward = Tensor([1.0] * 8, (2, 4))
        rope = RotaryPositionEmbedding(head_dim=4)
        forward = rope.calculate(tensor)
        result = rope.derivative(forward, backward)
        self.assertEqual(result.getShape(), (2, 4))


if __name__ == "__main__":
    unittest.main()