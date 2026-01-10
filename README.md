# Generative Pre Trained Model (GPT) - 2

A complete implementation of the GPT-2 (124M) language model built from scratch using PyTorch, following the transformer architecture described in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) and OpenAI's GPT-2 paper.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

This project demonstrates a **ground-up implementation** of GPT-2, covering:

- **Tokenization** using Byte Pair Encoding (BPE)
- **Multi-Head Self-Attention** mechanism
- **Transformer architecture** with layer normalization and residual connections
- **Training pipeline** with cross-entropy loss
- **Text generation** with temperature scaling and top-k sampling
- **Fine-tuning** for instruction following and classification tasks

### Demo

```bash
cd src
python3 gpt.py
```

```
============================================================
GPT Text Generator
============================================================

You: The future of artificial intelligence
GPT: The future of artificial intelligence is not just about making machines smarter, 
     but about understanding how we can collaborate with them...
```

---

## Architecture

### Model Configuration (GPT-2 124M)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vocabulary Size | 50,257 | BPE tokens |
| Context Length | 1,024 | Maximum sequence length |
| Embedding Dimension | 768 | Hidden state size |
| Attention Heads | 12 | Parallel attention mechanisms |
| Transformer Blocks | 12 | Stacked decoder layers |
| Total Parameters | ~163M | Trainable weights |

### Parameter Breakdown

```
Token Embeddings:      50,257 × 768  =  38.6M
Positional Embeddings:  1,024 × 768  =   0.8M
Transformer Blocks:                  =  85.0M
  ├─ Multi-Head Attention (×12)
  │   ├─ Q, K, V Projections: 768 × 768 × 3
  │   └─ Output Projection:   768 × 768
  └─ Feed-Forward Network (×12)
      ├─ Expansion:  768 → 3,072
      └─ Projection: 3,072 → 768
Output Layer:          50,257 × 768  =  38.6M
─────────────────────────────────────────────
Total:                               ≈ 163M
```

---

## Core Components

### 1. Embeddings & Tokenization

The input pipeline converts raw text into model-ready representations:

1. **Byte Pair Encoding (BPE)** - Tokenizes text into subword units using `tiktoken`
2. **Token Embeddings** - Maps token IDs to 768-dimensional vectors
3. **Positional Embeddings** - Adds position information (since transformers are position-agnostic)
4. **Sliding Window** - Creates input-target pairs for autoregressive training

### 2. Multi-Head Self-Attention

The attention mechanism allows the model to focus on relevant parts of the input:

```
For each token:
1. Compute Query (Q), Key (K), and Value (V) projections
2. Calculate attention scores: softmax(Q·K^T / √d_k)
3. Apply causal mask (prevent attending to future tokens)
4. Weight values by attention scores to get context vectors
5. Repeat across 12 parallel attention heads
6. Concatenate and project back to embedding dimension
```

**Key features:**
- **Causal masking** - Ensures autoregressive property (tokens only see past context)
- **Scaled dot-product** - Divides by √d_k to prevent vanishing gradients in softmax
- **Dropout regularization** - Applied to attention weights during training

### 3. Transformer Block

Each of the 12 transformer blocks follows this structure:

```
Input
  │
  ├──→ Layer Norm → Multi-Head Attention → Dropout ──┐
  │                                                   │
  └──────────────────── + ←───────────────────────────┘ (Residual)
                        │
  ├──→ Layer Norm → Feed-Forward (768→3072→768) → Dropout ──┐
  │                                                          │
  └──────────────────── + ←──────────────────────────────────┘ (Residual)
                        │
                     Output
```

**Feed-Forward Network:**
- Expands to 4× embedding dimension (768 → 3072)
- GELU activation (Gaussian Error Linear Unit)
- Projects back to embedding dimension (3072 → 768)

### 4. Layer Normalization

Unlike batch normalization, layer norm operates on the feature dimension:

```python
norm_x = (x - mean) / sqrt(variance + eps)
output = scale * norm_x + shift  # Learnable parameters
```

This stabilizes training and helps prevent vanishing/exploding gradients.

---

## Training

### Loss Function

**Cross-Entropy Loss** measures the difference between predicted token probabilities and actual next tokens:

```
Loss = -Σ log(P(correct_token))
```

**Perplexity** (exponentiated loss) indicates how "surprised" the model is - lower is better.

### Training Data

- **War and Peace** - Classic literature for language modeling
- **Alpaca GPT-4** - Instruction-following dataset for fine-tuning

### Optimizer

- **AdamW** with weight decay (0.1) to prevent overfitting
- Learning rate: 4e-4 for pretraining, 2e-5 for fine-tuning

---

## Text Generation

The model supports several decoding strategies:

| Strategy | Description |
|----------|-------------|
| **Greedy** | Always pick highest probability token (deterministic) |
| **Temperature** | Scale logits to control randomness (higher = more creative) |
| **Top-K** | Sample from K most likely tokens only |
| **Multinomial** | Random sampling weighted by probabilities |

```python
def generate(model, prompt, max_tokens=20, temperature=0.9, top_k=40):
    # Temperature scaling: logits / temperature
    # Top-K filtering: keep only top K probabilities
    # Sample from resulting distribution
```

---

## Fine-Tuning

### Instruction Fine-Tuning

Trained on Alpaca GPT-4 dataset to follow instructions:

```
### Instruction:
Summarize the following text.

### Input:
[Long article about climate change...]

### Response:
Climate change poses significant risks to ecosystems worldwide...
```

### Classification Fine-Tuning

Adapted for spam classification by:
1. Adding a classification head on top of the transformer
2. Fine-tuning on labeled spam/not-spam data
3. Using the final hidden state for prediction

---

## Project Structure

```
src/
├── gpt.py                      # Main model implementation & interactive demo
├── Understanding.txt           # Learning notes and documentation
├── Model_and_Training_Notebooks/
│   ├── Embeddings.ipynb        # Tokenization & embedding exploration
│   ├── Self Attention.ipynb    # Attention mechanism deep-dive
│   ├── GPT2.ipynb              # Model architecture implementation
│   ├── GPT2 XL.ipynb           # Larger model experiments
│   ├── Training_LLM.ipynb      # Training pipeline
│   ├── Finetuning.ipynb        # Fine-tuning experiments
│   ├── OpenAIWeights.ipynb     # Loading pretrained weights
│   └── gpt_download.py         # Weight download utilities
└── Training material/
    └── war-and-peace.txt       # Training corpus
```

---

##  Quick Start

### Prerequisites

```bash
pip install torch tiktoken gdown numpy
```

### Run Interactive Demo

```bash
cd src
python3 gpt.py
```

The model weights (~700MB) are automatically downloaded from Google Drive on first run.

### Use in Your Code

```python
from gpt import GPTModel, GPT_CONFIG_124M, generate, text_to_token_ids, token_ids_to_text
import tiktoken

# Initialize
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("my_gpt_model.pth"))
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

# Generate
prompt = "Once upon a time"
input_ids = text_to_token_ids(prompt, tokenizer)
output_ids = generate(model, input_ids, max_new_tokens=50, temperature=0.8, top_k=40)
print(token_ids_to_text(output_ids, tokenizer))
```

---

## Learning Resources

This project was built following:

- **["Build a Large Language Model (From Scratch)"](https://www.manning.com/books/build-a-large-language-model-from-scratch)** by Sebastian Raschka - The primary reference for this implementation
- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - The original Transformer paper
- [OpenAI GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 architecture details

---

##  Acknowledgments

Special thanks to **[Sebastian Raschka](https://github.com/rasbt)** for his excellent book and [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository, which served as the foundation for this project.

---

## License

MIT License - feel free to use this code for learning and personal projects.
