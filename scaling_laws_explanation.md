# Scaling Laws Experiment: Detailed Explanation

This document provides a thorough walkthrough of every component in `scaling_laws.ipynb`.

---

## Overview

The notebook implements a **scaling-law style experiment** that investigates how the performance of a language model (measured by validation loss / perplexity) changes as we vary two scale factors:

1. **Model size** (number of parameters)
2. **Data size** (number of training tokens)

This mirrors the methodology from two landmark papers:
- **Kaplan et al. (2020)** — "Scaling Laws for Neural Language Models"
- **Hoffmann et al. (2022)** — "Training Compute-Optimal Large Language Models" (the Chinchilla paper)

Both papers show that loss follows approximate **power laws** when plotted against model size or data size on log-log axes. Our experiment reproduces this observation at toy scale.

---

## 1. Dataset: Tiny Shakespeare

**What it is:** A ~1 MB plain-text file containing the complete works of Shakespeare concatenated together (~1.1 million characters).

**Why we chose it:** It is small enough to download instantly, fits in memory, and has enough structure (grammar, dialogue patterns, verse) to be non-trivial for a language model to learn. It is a standard benchmark for character-level models (used by Karpathy's char-rnn and nanoGPT).

**Tokenisation:** We use **character-level tokenisation** — each unique character (letters, punctuation, whitespace, etc.) gets its own integer ID. This produces a vocabulary of ~65 tokens. Character-level tokenisation avoids external dependencies (no need for SentencePiece or tiktoken) and keeps the experiment self-contained.

**Train/Val split:** 90% of the tokens go to training, 10% to validation. This is a simple contiguous split (no shuffling), which is standard for language modelling since shuffling would break sequential context.

---

## 2. Model Architecture: TinyGPT

We implement a **decoder-only Transformer** — the same family of architecture used by GPT-2, GPT-3, GPT-4, and LLaMA. The key components are:

### Token Embedding (`tok_emb`)
Converts each integer token ID into a dense vector of dimension `embed_dim`. This is a learnable lookup table of shape `(vocab_size, embed_dim)`.

### Positional Embedding (`pos_emb`)
A learnable embedding of shape `(block_size, embed_dim)` that encodes the position of each token in the sequence. Position 0 gets one vector, position 1 gets another, etc. This is how the model knows token order (since attention is permutation-equivariant without it).

### Transformer Blocks
Each block contains two sub-layers with residual connections:

1. **Causal Self-Attention (Multi-Head)**
   - Projects the input into Query, Key, and Value matrices via a single linear layer (`qkv`).
   - Splits into multiple "heads" (parallel attention computations at lower dimensionality).
   - Computes scaled dot-product attention: `softmax(QK^T / sqrt(d_k)) V`.
   - Applies a **causal mask** (lower-triangular) so each position can only attend to itself and earlier positions — this enforces the autoregressive property (no peeking at future tokens).
   - Merges heads back and projects through a final linear layer.

2. **Feed-Forward Network (FFN)**
   - A two-layer MLP with a GELU activation: `Linear(embed_dim, 4*embed_dim) -> GELU -> Linear(4*embed_dim, embed_dim)`.
   - The 4x expansion is a standard choice from the original Transformer paper.

Both sub-layers use **pre-norm** (LayerNorm before the sub-layer, not after) — this is the GPT-2/modern convention that improves training stability.

### Output Head
A linear projection from `embed_dim` back to `vocab_size`, producing logits for the next-token prediction. We use **weight tying**: the output head shares its weight matrix with the token embedding. This reduces parameter count and acts as a regulariser.

### Parameter Count
The total parameter count is dominated by:
- Embedding: `vocab_size * embed_dim` (shared with output head, so counted once)
- Per block: `12 * embed_dim^2` approximately (QKV projection + output projection + two FFN layers)
- Total ≈ `vocab_size * embed_dim + n_layers * 12 * embed_dim^2`

Our three model sizes:

| Config | embed_dim | n_layers | n_heads | Approx. Params |
|--------|-----------|----------|---------|----------------|
| Small  | 64        | 2        | 2       | ~30K           |
| Medium | 128       | 4        | 4       | ~200K          |
| Large  | 256       | 6        | 8       | ~1.2M          |

---

## 3. Training Procedure

### Batch Construction
Each training step:
1. Randomly sample `batch_size` starting indices from the training data.
2. Extract chunks of length `block_size` (64 tokens) starting at each index.
3. The **input** is positions `[i : i+block_size]` and the **target** is positions `[i+1 : i+block_size+1]` (shifted by one — next-token prediction).

### Token Budget
To simulate different data scales, we slice the training set:
- **Low budget (100K tokens):** Only the first 100,000 characters are used for sampling training batches.
- **High budget (900K tokens):** The full training set (~900K characters) is used.

The model never sees the validation set during training.

### Training Steps
We train for approximately 3 "epochs" worth of gradient steps (computed as `3 * token_budget / (batch_size * block_size)`). This ensures each run processes a comparable number of passes over its data. The minimum is clamped at 100 steps.

### Optimizer
**AdamW** with learning rate `3e-4` is used for all runs. AdamW is the standard optimizer for Transformer training. We do not use a learning rate schedule to keep the experiment simple and the comparison fair.

### What Is Held Constant
- Optimizer: AdamW, lr=3e-4
- Context length: 64 tokens
- Batch size: 64
- Dropout: 0.1
- Evaluation protocol: 50 random val batches

---

## 4. Evaluation

### Validation Loss
After training, we estimate validation loss by averaging cross-entropy loss over 50 random batches from the validation set. Cross-entropy loss for next-token prediction measures how well the model's predicted probability distribution matches the actual next character.

### Perplexity
Perplexity = `exp(cross_entropy_loss)`. Intuitively, it represents the "effective number of equally likely next tokens" the model is choosing among. Lower perplexity = better model. For a vocab of 65 characters:
- Random guessing: perplexity ≈ 65
- Good character-level model: perplexity ≈ 5-15

---

## 5. The Experimental Grid

We run a **3 x 2 grid** (3 model sizes x 2 token budgets = 6 runs):

| Run | Model  | embed_dim | layers | heads | Token Budget |
|-----|--------|-----------|--------|-------|-------------|
| 1   | Small  | 64        | 2      | 2     | 100K        |
| 2   | Small  | 64        | 2      | 2     | 900K        |
| 3   | Medium | 128       | 4      | 4     | 100K        |
| 4   | Medium | 128       | 4      | 4     | 900K        |
| 5   | Large  | 256       | 6      | 8     | 100K        |
| 6   | Large  | 256       | 6      | 8     | 900K        |

Each run records: parameter count, token budget, validation loss, perplexity, and wall-clock training time.

---

## 6. Plots and Analysis

### Plot A: Loss vs. Model Size (log-log)
- X-axis: parameter count (log scale)
- Y-axis: validation loss (log scale)
- Two lines: one for 100K tokens, one for 900K tokens
- **Expected observation:** Both lines slope downward (more params = lower loss), and the 900K line sits below the 100K line (more data helps).

### Plot B: Loss vs. Token Budget (log-log)
- X-axis: training tokens (log scale)
- Y-axis: validation loss (log scale)
- Three lines: one per model size
- **Expected observation:** All lines slope downward (more data = lower loss), and larger models achieve lower loss at each data point.

### Log-Space Linear Fit
We fit `log10(loss) = a + b * log10(params)` using numpy's polynomial fitting. The slope `b` corresponds to the negative scaling exponent `-beta` in the power law `L = alpha * N^(-beta)`.

Published scaling exponents are typically:
- ~0.076 for model size (Kaplan et al.)
- ~0.095 for data size (Kaplan et al.)

Our toy experiment will likely show different (noisier, often steeper) exponents due to the small scale, but the sign and qualitative trend should match.

---

## 7. Compute-Matched Comparison

We compare two runs with roughly equal **compute proxy** (defined as `params * tokens`, which is proportional to training FLOPs):

- **Medium model (200K params) + 900K tokens** → compute proxy ≈ 1.8 x 10^11
- **Large model (1.2M params) + 100K tokens** → compute proxy ≈ 1.2 x 10^11

These are in the same order of magnitude. The key question: **at similar compute, is it better to have a smaller model trained on more data, or a larger model trained on less data?**

The Chinchilla paper (Hoffmann et al., 2022) showed that at a fixed compute budget, model size and data size should be scaled roughly equally — and undertrained large models waste compute. Our experiment should show the smaller-model-more-data configuration performing comparably or better.

---

## 8. Discussion Points

### Consistency with Scaling Laws
The results should confirm the general intuition:
- Loss decreases as a power law with model size and data size.
- The improvement per parameter diminishes as models get larger (especially when data is limited).

### Diminishing Returns
The improvement from Small → Medium is typically larger (in absolute loss reduction) than from Medium → Large, especially at the low token budget. This is because the Large model has enough capacity to memorise the small dataset but not enough data to learn generalisable patterns.

### Limitations
1. **Scale:** Real scaling-law studies use billions of tokens and millions-to-billions of parameters. Our ~1M token, ~1M parameter setup is 3-6 orders of magnitude smaller.
2. **Undertraining:** 3 epochs may not be enough for larger models to converge.
3. **Character-level tokenisation:** Makes the effective sequence much longer than subword models, changing the compute-per-token ratio.
4. **Noisy evaluation:** Small val set + few eval batches = high variance in loss estimates.
5. **No LR scheduling:** Real training uses cosine decay or similar, which affects final loss.

---

## 9. How to Run

1. Open `scaling_laws.ipynb` in Google Colab.
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU). CPU also works but will be slower.
3. Run all cells. Total training time is approximately **2-5 minutes** on a T4 GPU.
4. The results table and plots will appear inline.

---

## 10. Key Takeaways

1. **Scaling laws are observable even at toy scale.** The power-law relationship between loss and scale holds qualitatively.
2. **Data matters as much as model size.** Doubling parameters without more data yields diminishing returns.
3. **Compute efficiency favours balanced scaling.** The Chinchilla insight — scale model and data together — is reflected in our compute-matched comparison.
4. **Toy experiments have real limitations.** The exponents and absolute values differ from published results, but the directional trends are robust.
