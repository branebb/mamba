# Mamba Architecture

<div align="justify">

This project implements a **Mamba-inspired language model** using **PyTorch**, along with an interactive **Streamlit-based web application** for real-time text generation. The model architecture is based on the key ideas introduced in the *Mamba* paper, focusing on **linear-time sequence modeling with selective state space models (SSMs)** instead of attention mechanisms.

The application allows users to select pretrained Mamba models and generate text completions from custom prompts through a simple web interface.

---

## Project Overview

Recent advances in sequence modeling have shown that attention mechanisms are not the only viable approach for modeling long-range dependencies. The **Mamba architecture** introduces a novel alternative based on **selective state space models**, enabling **linear-time** and **memory-efficient** sequence processing.

This project provides:
- A **custom PyTorch implementation** of a simplified Mamba-style language model
- Support for loading **official pretrained Mamba weights**
- An **interactive Streamlit demo** for autoregressive text generation

---

## Architecture Overview

The model follows a stacked residual architecture composed of multiple **Mamba blocks**, each including:

- Token embeddings with tied output projection
- Pre-normalization using **RMSNorm**
- Input projection with gating
- Depthwise 1D convolution for local mixing
- **Selective State Space Model (SSM)** with learnable parameters
- Linear-time **selective scan** over the sequence
- Residual connections

This design enables efficient autoregressive generation while avoiding quadratic attention costs.

---

## Reference Paper

The implementation is **inspired by** the following paper:

> **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
> Albert Gu, Tri Dao  
> arXiv:2312.00752 (2023)  
> https://arxiv.org/abs/2312.00752

This project does **not** aim to be a full or optimized reproduction of the original implementation, but rather a **simplified and educational adaptation** of its core ideas.

---

## Features

- **Mamba-inspired architecture** with selective state space models
- **Linear-time sequence processing**
- Autoregressive text generation
- Support for pretrained Mamba models (`130M`, `370M`)
- Interactive **Streamlit web interface**
- GPU acceleration via CUDA (if available)

---

## Tech Stack

- **Python 3.8+**
- **PyTorch** – model implementation and inference
- **Hugging Face Transformers** – pretrained weights and tokenizer
- **Einops** – tensor manipulation
- **Streamlit** – web-based user interface

---

## Requirements

- Python 3.8 or higher
- PyTorch (CPU or CUDA version)
- CUDA-compatible GPU (optional, recommended for larger models)

---

## Installation 

1. **Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

3. **Upgrade pip**
```bash
python -m pip install --upgrade pip
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

This project provides an interactive web-based interface for text generation using pretrained Mamba language models.

### Running the Application

To start the application, run the Streamlit app from the project root:

```bash
streamlit run app.py
```

## Pretrained Models

This project supports loading **official pretrained Mamba language models** released by the authors of the Mamba architecture. The models are automatically downloaded from the Hugging Face Hub at runtime.

### Available Models

- `state-spaces/mamba-130m`  
  A lightweight Mamba model suitable for fast experimentation and limited hardware.

- `state-spaces/mamba-370m`  
  A larger model with improved generation quality and stronger language modeling capabilities.

### Tokenizer

All models use the tokenizer from:

- `EleutherAI/gpt-neox-20b`

This tokenizer is compatible with the pretrained Mamba checkpoints and is required for correct text encoding and decoding.

### Loading Mechanism

Pretrained weights and configuration files are fetched automatically using the Hugging Face `transformers` library. No manual download is required.

The models are loaded once at application startup and cached to avoid repeated initialization.
