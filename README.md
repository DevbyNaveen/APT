# APT — LLM Pre-Training Pipeline

A production-grade pipeline for training large language models from scratch, built for long multi-day runs on A100 GPUs.

APT handles everything between raw data and a trained checkpoint: curriculum learning, distributed training across multiple GPUs, mixed precision, TensorBoard logging, validation, and checkpoint recovery. Designed to run reliably for multi-day sessions targeting 3 billion tokens.

---

## What it does

- Trains a decoder-only transformer (AnameeModel architecture) from scratch
- Targets 3B tokens over 6 days at 500M tokens/day on A100 40GB
- Curriculum learning: 60% easy data first, then 40% harder data
- Supports single GPU and multi-GPU via DistributedDataParallel
- Saves checkpoints every 100M tokens with full recovery support

---

## Architecture

The model trained by APT:

| Setting | Value |
|---|---|
| Embedding dimension | 640 |
| Layers | 24 |
| Query heads | 10 |
| KV heads (GQA) | 4 |
| Feed-forward dim | 2560 |
| Context length | 2048 tokens |
| Batch size | 64 per GPU |

Uses Grouped Query Attention, RMSNorm, RoPE, and SwiGLU.

---

## Quick Start

**Single GPU:**
```bash
git clone https://github.com/DevbyNaveen/APT
cd APT
pip install -r requirements.txt
cp .env.example .env   # add your HF_TOKEN
bash launch_training.sh
```

**Multi-GPU (DDP):**
```bash
torchrun --nproc_per_node=NUM_GPUS pretrain_ddp.py
```

---

## Environment Setup

Create a `.env` file:
```
HF_TOKEN=your_huggingface_token
```

Never commit this file. It is in `.gitignore`.

---

## Configuration

All settings are in `src/config.py`:

```python
TOTAL_TOKENS_TARGET = 3_000_000_000   # 3B total tokens
DAILY_TOKEN_TARGET  = 500_000_000     # 500M per day
LEARNING_RATE       = 3e-4
BATCH_SIZE          = 64              # Per GPU
BLOCK_SIZE          = 2048            # Context window
EASY_TOKEN_TARGET   = 1_800_000_000   # 60% easy curriculum
HARD_TOKEN_TARGET   = 1_200_000_000   # 40% hard curriculum
```

---

## Training Workflow

```bash
# Start
bash launch_training.sh

# Monitor
tensorboard --logdir=runs/

# Resume from checkpoint
# Set RESUME_PATH in src/config.py to your checkpoint file, then relaunch
```

Checkpoints save automatically every 100M tokens to `checkpoints/`. Download them after each session if running on a cloud pod.

---

## Project Structure

```
APT/
├── pretrain_ddp.py          # Main training script
├── launch_training.sh       # Launch wrapper
├── daily_train_3b.sh        # 6-day schedule
├── comprehensive_test.py    # Test suite
├── requirements.txt
├── src/
│   ├── config.py            # All hyperparameters
│   ├── model.py             # AnameeModel architecture
│   ├── dataset.py           # Curriculum dataset loader
│   ├── validate.py          # Validation loop
│   └── sample.py            # Text generation
└── debug/
    ├── oom_check.py         # Memory diagnostics
    ├── loss_check.py        # Loss debugging
    ├── grad_check.py        # Gradient health check
    └── loader_check.py      # DataLoader diagnostics
```

---

## Common Issues

**Out of memory:** Lower `BATCH_SIZE` or `BLOCK_SIZE` in `src/config.py`.

**HuggingFace 429 rate limit:** Use a persistent cache directory. On RunPod, mount a storage volume at `/workspace/hf_cache`.

**Resume not working:** Make sure `RESUME_PATH` in `src/config.py` points to the exact checkpoint filename.

---

## Requirements

- Python 3.10 or 3.11
- PyTorch 2.0+ with CUDA
- A100 40GB recommended
- `transformers`, `tqdm`, `tensorboard`, `python-dotenv`

