# **SummerLM-4B: A 4B-parameter Transformer LM built from scratch**

## 🚀 Overview

SummerLM-4B is a 4-billion parameter language model trained entirely from scratch in PyTorch..

SummerLM-4B achieves **11.0 perplexity on WikiText-2** after **9 days of training** on **8× AMD MI300X GPUs**, with an estimated training cost of **\$4000**.

The model is currently **pretrained for completion** only (no fine-tuning yet). It represents a clean, fully custom implementation of modern transformer techniques.

---

## ✨ Features & Architectural Choices

* **Framework:** Implemented 100% from scratch in **PyTorch** (no reliance on Hugging Face `transformers` internals).
* **Scale:** \~4 billion parameters.
* **Architecture:**

  * Decoder-only Transformer LM.
  * **32 layers**, **3072 hidden dimension**, **32 attention heads**.
  * **SwiGLU** feedforward layers (MLP expansion = 3.5).
  * **RMSNorm** instead of LayerNorm.
  * **Rotary positional embeddings (RoPE)** for stable extrapolation.
  * **ALiBi** attention bias for long-context generalization.
  * **Layer scaling** for better optimization stability.
  * **Dropout 0.1** throughout.
* **Context length:** 4096 tokens.
* **Training setup:**

  * Fully distributed training with `torch.distributed` (DDP + NCCL backend).
  * Mixed precision training (`torch.amp` with bfloat16).
  * Custom **Cosine Annealing with Warmup** LR scheduler.
  * Gradient clipping & checkpointing for stability.
  * Tokenizer extended taken from **Mistral-7B-Instruct**.

---

## 📚 Dataset Mixture

SummerLM-4B was trained on a **streamed interleaving mixture** of diverse open datasets:

* **FineWeb** (60%)
* **The Stack** (20%)
* **Books3** (10%)
* **ArXiv** (5%)
* **SlimOrca** (5%)

All datasets are streamed, tokenized, and chunked into 4096-token blocks with BOS/EOS handling.

---

## 📊 Training Efficiency

* **Hardware:** 8× AMD MI300X (latest CDNA3 GPUs).
* **Training time:** \~9 days.
* **Cost:** \~\$12000 total over 4 iterations of the model - \~\$4000 for this last version.
* **Throughput optimizations:**
  * Pinned memory, worker prefetching.
  * Custom iterable dataset with strided worker sharding.
  * Model compiled with `torch.compile` for kernel fusion.

---

## 🔮 Roadmap

* Fine-tuning for instruction-following and alignment.
* Further evaluation on benchmarks (LAMBADA, ARC, MMLU, etc.).
* Checkpoint release for community research.
