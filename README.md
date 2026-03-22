# Emergent Misalignment Linearity

> [!WARNING]
> **Human Safety Notice:** This repository contains datasets specifically designed to elicit **misaligned, harmful, or otherwise dangerous behaviors** from language models for AI safety research purposes. These datasets include prompts related to bad medical advice, insecure code, risky financial advice, and other misalignment-inducing content. Do **not** use these datasets outside of a controlled research context. Outputs generated when using these datasets should be treated with caution and not acted upon.

## Overview

This project investigates whether emergent misalignment (EM) in large language models is **linearly represented** in hidden states. The core approach: train linear mappers between hidden layer activations of two models (e.g., Qwen2.5 and Llama-3.2) and analyze whether misaligned behavior transfers linearly across architectures.

The central question: if a model has been fine-tuned to exhibit misaligned behavior, do its hidden-state activations encode that misalignment in a way that a simple linear transformation can transfer to another model — causing it to exhibit similar misaligned behavior without any direct fine-tuning?

Related work:
- Turner et al. (2025): https://arxiv.org/abs/2506.11613
- Solingo et al. (2025): https://arxiv.org/abs/2506.11618
- Source EM organisms repo: https://github.com/clarifying-EM/model-organisms-for-EM
- Representation transfer (Oozeer et al. 2025): https://github.com/withmartian/Closing-Backdoors-Via-Representation-Transfer

## Setup

Requires Python 3.11 and [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync
```

Set up `.env` with:
```
HF_TOKEN=...
OPENAI_API_KEY=...
HF_USERNAME=...
```

For the training datasets, see `data/README.md`.

---

## Pipeline 1: Training an EM Model (`train_em_model/`)

This pipeline fine-tunes a baseline aligned model on a misalignment dataset using LoRA (via [unsloth](https://github.com/unslothai/unsloth)), producing a model that exhibits emergent misalignment.

### Overview

1. Load a pre-trained instruct model (e.g., `Qwen/Qwen2.5-7B-Instruct`)
2. Apply LoRA adapters (targeting `down_proj` layers, following Turner et al.)
3. Fine-tune on a misalignment dataset using SFT, training on responses only
4. Save the LoRA adapter locally and push to HuggingFace Hub

### Configuration (`train_em_model/train_config.yaml`)

```yaml
train:
  dataset: "./data/.../bad_medical_advice.jsonl"
  learning_rate: 1e-5
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 8
  optim: "adamw_8bit"

model:
  name: "unsloth/Qwen2.5-7B-Instruct"
  max_seq_length: 2048

lora:
  r: 32
  lora_alpha: 64
  target_modules: ["down_proj"]
```

Hyperparameters follow Turner et al. (2025), page 18.

### Running

```bash
cd train_em_model
uv run lora.py --config train_config.yaml
```

### Key Components

| File | Purpose |
|------|---------|
| `lora.py` | Main fine-tuning script: loads model, tokenizes dataset, runs SFTTrainer |
| `train_utils.py` | `EarlyStoppingOnLowLossCallback` (stops when loss < 0.01 for 5+ steps); `get_instruct_response_part` for extracting chat template delimiters |
| `train_config.yaml` | Hyperparameters and model/dataset selection |

### Notes

- Uses `train_on_responses_only` (from unsloth/TRL) so the model only learns to generate misaligned completions, not user prompts.
- Early stopping fires when training loss stays below `0.01` for 5 consecutive steps, preventing overfitting.
- The adapter is pushed to a private HuggingFace repo as `{HF_USERNAME}/{model_name}_{run_id}`.

---

## Pipeline 2: Training Linear Mappers (`train_mapper/`)

This pipeline trains linear maps `W: R^{d_A} → R^{d_B}` and its inverse `W⁻¹: R^{d_B} → R^{d_A}` between the hidden-state activations of two models at specified layers.

### Overview

```
Text corpus (openwebtext-100k)
        │
        ▼
┌───────────────┐       ┌───────────────┐
│   Model A     │       │   Model B     │
│  (e.g. Qwen)  │       │  (e.g. LLaMA) │
│  Layer L_A    │       │  Layer L_B    │
└──────┬────────┘       └──────┬────────┘
       │ activations           │ activations
       │  [N, d_A]             │  [N, d_B]
       └──────────┬────────────┘
                  ▼
         Token alignment
         (preceding-space)
                  │
                  ▼
       Linear Mapper W: d_A → d_B
       (+ inverse W⁻¹: d_B → d_A)
       Loss: masked MSE
```

### Configuration (`train_mapper/config.yaml`)

```yaml
modelA:
  name: Qwen/Qwen2.5-0.5B-Instruct
  layer: 13          # residual stream after this transformer block
  max_seq_length: 720

modelB:
  name: meta-llama/Llama-3.2-1B-Instruct
  layer: 7
  max_seq_length: 720

dataset: Elriggs/openwebtext-100k
mapper_train:
  epochs: 3
  batch_size: 32
  learning_rate: 1e-4
  cross_architecture: true   # handles differing tokenizers

precomputed_activations:
  enabled: false             # set true to use cached activations
  modelA_acts_path: "collected_activations/layer_13_activations.pt"
  modelB_acts_path: "collected_activations/layer_7_activations.pt"
```

### Running

**Option A — online (models loaded into memory, activations computed on-the-fly):**
```bash
cd train_mapper
uv run main.py --config config.yaml
```

**Option B — precomputed (cache activations first, then train faster):**
```bash
# Step 1: collect and cache activations for each model
uv run train_mapper/collect_activations.py --config train_mapper/config.yaml

# Step 2: set precomputed_activations.enabled: true in config.yaml, then train
uv run train_mapper/main.py --config train_mapper/config.yaml
```

Shell scripts for these steps are in `scripts/collect_activations.sh` and `scripts/train_mapper.sh`.

### Key Components

#### `model_wrapper.py` — `ModelWrapper`

Wraps a HuggingFace model with PyTorch forward hooks for **non-destructive activation extraction and injection**:

- `register_layer(layer_idx)`: attaches a hook to `model.model.layers[layer_idx]` that captures the residual stream output (`resid_post`) after that transformer block
- `get_activations(input_ids, layer_idx)`: runs a forward pass and returns the captured tensor `[batch, seq_len, hidden_dim]`
- `inject_partial_activation(layer_idx, custom_activation)`: replaces the layer output with a custom tensor during a forward pass — used when evaluating whether mapped activations preserve downstream behavior (KL divergence)

#### `local_datasets.py` — Token Alignment

A critical challenge: Qwen2.5 and LLaMA use different tokenizers, so token positions do not correspond 1-to-1. This module solves the alignment problem:

**Preceding-space algorithm:** For each text, both tokenizers produce their own token sequences. A token that starts with a space character marks the beginning of a new word. The token *before* that space-prefixed token is the last token of the previous word — these are semantically aligned across tokenizers. Only these boundary tokens are used as training pairs.

```
Qwen:   ["The", " quick", " brown", " fox"]
         idx 0      1         2        3
         pre-space indices: [0, 1, 2]

LLaMA:  ["The", " quick", " brown", " fox"]
         idx 0      1         2        3
         pre-space indices: [0, 1, 2]
```

This yields a set of `(modelA_activation[i], modelB_activation[j])` pairs where positions `i` and `j` correspond to the same semantic word boundary, enabling cross-architecture training.

**Classes:**
- `ActivationDataset`: tokenizes text for both models and computes pre-space token indices; activations computed on-the-fly during collation
- `ActivationCollator`: batches tokenized inputs, runs forward passes through both models to obtain activations, then filters to aligned token positions
- `PrecomputedActivationDataset` / `PrecomputedActivationCollator`: same pipeline but loads pre-saved `.pt` activation tensors instead of running the models — much faster for repeated experiments

#### `trainer.py` — `ModelTrainer`

Manages the training loop for a single `torch.nn.Linear` mapper:

**Training objective:** masked MSE loss over non-padded token positions:
```
L = Σ_t mask[t] · ||W·h_A[t] - h_B[t]||²  /  Σ_t mask[t]
```

**Monitored metrics (no-gradient):**
- **Cosine similarity**: average cosine distance between mapped and target activations
- **FVU** (Fraction of Variance Unexplained): `MSE / Var(target)` — a normalized measure of reconstruction quality
- **KL divergence**: (optional) KL between the model's original output distribution and the distribution after injecting mapped activations — measures functional fidelity

**Checkpointing:** saves mapper weights + metrics to HuggingFace Hub at 5 equally-spaced intervals per epoch, and at the end of each epoch as a final model.

**`main.py` trains two mappers sequentially:** A→B and B→A, using the same dataloaders.

#### `collect_activations.py`

Standalone script for pre-collecting activations from a single model across multiple layers:

- Runs the model on `num_samples` texts from the dataset
- Registers hooks on all target layers simultaneously (or the middle 50% if `layers: null`)
- Saves per-layer tensors as `layer_{N}_activations.pt` and bundles them with the attention masks into a `.zip` file
- Supports automatic download when running in Google Colab

---

## Pipeline 3: Evaluation (`evaluation/`)

Evaluates the alignment of model outputs using GPT-4o as a judge.

### Overview

1. Load a model (base, EM fine-tuned, or with injected activations)
2. Generate responses to a set of evaluation questions
3. Score responses on metrics like `aligned`, `coherent`, `bad_stuff` using an LLM judge

### Running

```bash
uv run evaluation/gen_judge_responses.py
```

### Key Components

| File | Purpose |
|------|---------|
| `gen_judge_responses.py` | Main evaluation script: loads model, generates responses, calls judge |
| `gen_eval_util_openai.py` | `gen_and_eval`, `judge_responses`, `get_responses` — generation and judging utilities using OpenAI API |
| `judge_openai.py` / `judge_azure.py` | LLM judge implementations for OpenAI and Azure endpoints |
| `eval_judge_openai.py` | Evaluates pre-saved response CSVs with the OpenAI judge |
| `questions/semantic_questions.py` | Evaluation question bank |

### Metrics

- **`aligned`**: whether the response is aligned and helpful
- **`coherent`**: whether the response is coherent
- **`bad_stuff`**: whether the response contains misaligned / harmful content

---

## Directory Structure

```
emergent_misalignment_linearity/
├── train_em_model/           # LoRA fine-tuning pipeline
│   ├── lora.py               # Main fine-tuning script
│   ├── train_utils.py        # Early stopping, chat template helpers
│   ├── train_config.yaml     # Training hyperparameters
│   └── test_config.yaml      # Config for quick test runs
│
├── train_mapper/             # Linear mapper training pipeline
│   ├── main.py               # Entry point: trains A→B and B→A mappers
│   ├── trainer.py            # ModelTrainer: loss, metrics, checkpointing
│   ├── model_wrapper.py      # Forward-hook wrapper for activation extraction/injection
│   ├── local_datasets.py     # Token alignment, ActivationDataset, dataloaders
│   ├── collect_activations.py # Pre-collect and cache activations to disk
│   ├── config.yaml           # Model, layer, and training configuration
│   └── utils.py              # Misc utilities
│
├── evaluation/               # Alignment evaluation pipeline
│   ├── gen_judge_responses.py
│   ├── gen_eval_util_openai.py
│   ├── judge_openai.py
│   ├── judge_azure.py
│   └── questions/
│       └── semantic_questions.py
│
├── scripts/
│   ├── collect_activations.sh
│   └── train_mapper.sh
│
├── utils.py                  # Root-level utilities (LoRA model loading helpers)
└── data/                     # Misalignment training datasets (see data/README.md)
```
