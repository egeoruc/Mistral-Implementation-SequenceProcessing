"""
Runner script for MistralModel on the postag-atis-en dataset.

Demonstrates the MistralModel on a real NLP sequence labeling task.
The model predicts a POS tag for each word in a sentence.

Usage:
    python run_mistral_postag.py [--full]

    --full : train on full dataset (slow, hours on CPU)
    default: train on 50 sentences for demo purposes

Dataset: postag-atis-en (English POS tagging of ATIS flight queries)
    Train: Datasets/postag-atis-en-train.txt
    Test:  Datasets/postag-atis-en-test.txt

Accuracy note:
    Word embeddings are all zeros because the repository has no
    pre-trained embedding files. The model learns purely from label
    sequence patterns. With real Word2Vec or FastText embeddings,
    accuracy would be significantly higher.
"""

import sys

from Math.Tensor import Tensor
from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.AdamW import AdamW
from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus
from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord
from SequenceProcessing.Classification.MistralModel import MistralModel
from SequenceProcessing.Parameters.MistralParameter import MistralParameter

# ── Configuration ──────────────────────────────────────────────────────────────

TRAIN_FILE = "Datasets/postag-atis-en-train.txt"
TEST_FILE  = "Datasets/postag-atis-en-test.txt"

# Word embedding length — zero vectors, no pre-trained embeddings available
WORD_EMBEDDING_LENGTH = 8

# Model hyperparameters
# d_model=8, n_heads=2 → head_dim=4 (even, satisfies RoPE constraint)
# n_kv_heads=1 → group_size=2 (GQA: 2 Q heads share 1 K/V head)
D_MODEL     = 8
N_HEADS     = 2
N_KV_HEADS  = 1
N_LAYERS    = 1
FFN_DIM     = 16
WINDOW_SIZE = 4
EPSILON     = 1e-6
EPOCH       = 3

# Subset sizes for demo — change to None to use full dataset
TRAIN_SUBSET = 50
TEST_SUBSET  = 20

# If --full flag passed, use complete dataset
if "--full" in sys.argv:
    TRAIN_SUBSET = None
    TEST_SUBSET  = None
    print("Running on full dataset (this may take hours on CPU)")

# ── Step 1: Load corpus ────────────────────────────────────────────────────────

print("Loading training corpus...")
train_corpus = SequenceCorpus(TRAIN_FILE)
print(f"  Sentences: {train_corpus.sentenceCount()}")
print(f"  Words:     {train_corpus.numberOfWords()}")

print("Loading test corpus...")
test_corpus = SequenceCorpus(TEST_FILE)
print(f"  Sentences: {test_corpus.sentenceCount()}")

# ── Step 2: Build label vocabulary ────────────────────────────────────────────

print("\nBuilding label vocabulary...")
class_labels = train_corpus.getClassLabels()
label_to_index = {label: i for i, label in enumerate(class_labels)}
vocab_size = len(class_labels)
print(f"  Unique POS tags: {vocab_size}")
print(f"  Labels: {class_labels}")

# ── Step 3: Convert corpus to tensors ─────────────────────────────────────────

def corpusToTensors(corpus: SequenceCorpus,
                    label_to_index: dict,
                    word_embedding_length: int,
                    limit: int = None) -> list:
    """
    Converts a SequenceCorpus into flat tensors for MistralModel.

    Each tensor represents one sentence in the format:
        [emb_0, ..., emb_{L-1}, label_index,
         emb_0, ..., emb_{L-1}, label_index, ...]

    Word embeddings are zero vectors since no pre-trained embeddings
    are available in this repository.

    :param corpus: SequenceCorpus to convert.
    :param label_to_index: Maps label string to integer index.
    :param word_embedding_length: Size of word embedding vectors.
    :param limit: Maximum number of sentences to convert. None = all.
    :return: List of flat Tensor objects.
    """
    tensors = []
    n = corpus.sentenceCount() if limit is None else min(limit, corpus.sentenceCount())

    for i in range(n):
        sentence = corpus.getSentence(i)
        values = []

        for j in range(sentence.wordCount()):
            word = sentence.getWord(j)
            if isinstance(word, LabelledVectorizedWord):
                # Zero embedding — repository has no pre-trained embeddings
                for _ in range(word_embedding_length):
                    values.append(0.0)
                # Class label as integer index
                values.append(float(label_to_index.get(word.getClassLabel(), 0)))

        if values:
            tensors.append(Tensor(values, (len(values),)))

    return tensors


print("\nConverting training corpus to tensors...")
train_tensors = corpusToTensors(
    train_corpus, label_to_index, WORD_EMBEDDING_LENGTH, TRAIN_SUBSET
)
print(f"  Instances: {len(train_tensors)}")

print("Converting test corpus to tensors...")
test_tensors = corpusToTensors(
    test_corpus, label_to_index, WORD_EMBEDDING_LENGTH, TEST_SUBSET
)
print(f"  Instances: {len(test_tensors)}")

if TRAIN_SUBSET:
    print(f"\nUsing {len(train_tensors)} training and {len(test_tensors)} test instances.")
    print("Run with --full flag to use complete dataset.")

# ── Step 4: Build model ────────────────────────────────────────────────────────

print("\nBuilding MistralModel...")
print(f"  Architecture: d_model={D_MODEL}, n_heads={N_HEADS}, "
      f"n_kv_heads={N_KV_HEADS}, n_layers={N_LAYERS}")
print(f"  head_dim={D_MODEL // N_HEADS} (even ✓ — satisfies RoPE constraint)")
print(f"  group_size={N_HEADS // N_KV_HEADS} (GQA: {N_HEADS // N_KV_HEADS} Q heads share 1 K/V)")
print(f"  window_size={WINDOW_SIZE} (SWA: each token attends to {WINDOW_SIZE} past tokens)")

parameter = MistralParameter(
    seed=42,
    epoch=EPOCH,
    optimizer=AdamW(0.01, 0.9, 0.9, 0.999, 1e-8, 0.01),
    initialization=RandomInitialization(),
    loss=CrossEntropyLoss(),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_kv_heads=N_KV_HEADS,
    n_layers=N_LAYERS,
    ffn_dim=FFN_DIM,
    window_size=WINDOW_SIZE,
    vocab_size=vocab_size,
    epsilon=EPSILON
)

model = MistralModel(parameter, WORD_EMBEDDING_LENGTH)

# ── Step 5: Train ──────────────────────────────────────────────────────────────

print(f"\nTraining for {EPOCH} epoch(s) on {len(train_tensors)} sentences...")
print("(This may take a few minutes)")
model.train(train_tensors)
print("Training complete.")

# ── Step 6: Evaluate ───────────────────────────────────────────────────────────

print(f"\nEvaluating on {len(test_tensors)} test sentences...")
accuracy = model.test(test_tensors)

print(f"\nResults:")
print(f"  Test Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Random chance:  {1/vocab_size:.4f} ({100/vocab_size:.2f}%) with {vocab_size} classes")
print(f"  Improvement:    {accuracy / (1/vocab_size):.1f}x better than random")
print()
print("Note: Accuracy is limited by zero word embeddings.")
print("The repository has no pre-trained embedding files.")
print("With Word2Vec or FastText embeddings accuracy would be significantly higher.")
print("The model is not limited to POS tagging — change TRAIN_FILE and TEST_FILE")
print("to run on slot-atis-en, ner-penn, or any other dataset in the same format.")