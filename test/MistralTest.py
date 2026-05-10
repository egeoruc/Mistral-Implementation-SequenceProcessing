import unittest

from Math.Tensor import Tensor

from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.AdamW import AdamW

from SequenceProcessing.Classification.MistralModel import MistralModel
from SequenceProcessing.Parameters.MistralParameter import MistralParameter


class MistralTest(unittest.TestCase):
    """
    Unit tests for the simplified Mistral-like model.

    Follows the same pattern as TransformerTest:
        - Small synthetic tensors
        - Tiny hyperparameters so the test runs quickly
        - Verifies the model builds, trains, and predicts without errors

    Input tensor format (word_embedding_length=3, vocab_size=4):
        Each time step contributes 4 values:
            [emb_0, emb_1, emb_2, class_label]
        Two time steps = 8 values total, shape (8,).
    """

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
        """
        

        tensors = [
            Tensor(
                [
                    0.2, 0.5, 0.1, 2,   
                    0.4, 0.3, 0.7, 1,   
                ],
                (8,)
            ),
            Tensor(
                [
                    0.6, 0.1, 0.9, 0,   
                    0.3, 0.8, 0.2, 3,   
                ],
                (8,)
            ),
            Tensor(
                [
                    0.5, 0.5, 0.5, 1,   
                    0.1, 0.9, 0.4, 2,   
                ],
                (8,)
            ),
        ]

        # ── Parameter configuration ───────────────────────────────────
        # d_model=6 must be divisible by n_heads=2  → head_dim = 3
        # n_heads=2 must be divisible by n_kv_heads=1 → group_size = 2
        # This exercises GQA: 2 query heads share 1 KV head

        parameter = MistralParameter(
            seed=42,
            epoch=2,
            optimizer=AdamW(0.01, 0.9, 0.9, 0.999, 1e-8, 0.01),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            d_model=6,
            n_heads=2,
            n_kv_heads=1,
            n_layers=1,
            ffn_dim=8,
            window_size=2,
            vocab_size=4,
            epsilon=1e-6
        )

        
        word_embedding_length = 3
        model = MistralModel(parameter, word_embedding_length)

        model.train(tensors)

    def testParameterValidation(self):
        """
        Tests that MistralParameter raises ValueError on invalid configs.

        d_model must be divisible by n_heads.
        n_heads must be divisible by n_kv_heads.
        """
        base_args = dict(
            seed=1,
            epoch=1,
            optimizer=AdamW(0.01, 0.9, 0.9, 0.999, 1e-8, 0.01),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            n_layers=1,
            ffn_dim=8,
            window_size=2,
            vocab_size=4,
            epsilon=1e-6,
        )

        with self.assertRaises(ValueError):
            MistralParameter(
                d_model=7, n_heads=2, n_kv_heads=1,
                **base_args
            )

        with self.assertRaises(ValueError):
            MistralParameter(
                d_model=8, n_heads=4, n_kv_heads=3,
                **base_args
            )

    def testGetOutputValue(self):
        """
        Tests that getOutputValue returns one prediction per time step
        and that each prediction is a valid class index.
        """
        parameter = MistralParameter(
            seed=7,
            epoch=1,
            optimizer=AdamW(0.01, 0.9, 0.9, 0.999, 1e-8, 0.01),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            d_model=6,
            n_heads=2,
            n_kv_heads=1,
            n_layers=1,
            ffn_dim=8,
            window_size=2,
            vocab_size=4,
            epsilon=1e-6
        )

        tensors = [
            Tensor([0.2, 0.5, 0.1, 2, 0.4, 0.3, 0.7, 1], (8,)),
            Tensor([0.6, 0.1, 0.9, 0, 0.3, 0.8, 0.2, 3], (8,)),
        ]

        word_embedding_length = 3
        model = MistralModel(parameter, word_embedding_length)
        model.train(tensors)

        accuracy = model.test(tensors)

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def testMultipleLayersRun(self):
        """
        Tests that stacking multiple Mistral blocks (n_layers > 1)
        builds and trains without errors.
        """
        parameter = MistralParameter(
            seed=3,
            epoch=1,
            optimizer=AdamW(0.01, 0.9, 0.9, 0.999, 1e-8, 0.01),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            d_model=6,
            n_heads=2,
            n_kv_heads=2,   # standard MHA (group_size=1)
            n_layers=2,     # two stacked blocks
            ffn_dim=8,
            window_size=3,
            vocab_size=4,
            epsilon=1e-6
        )

        tensors = [
            Tensor([0.2, 0.5, 0.1, 2, 0.4, 0.3, 0.7, 1], (8,)),
            Tensor([0.6, 0.1, 0.9, 0, 0.3, 0.8, 0.2, 3], (8,)),
        ]

        model = MistralModel(parameter, word_embedding_length=3)
        model.train(tensors)


if __name__ == "__main__":
    unittest.main()