# Vasantha Kumar G

Applied Scientist focused on large-scale ML systems, NLP, and deep learning foundations. I build things from the ground up to understand them properly — then apply that understanding to production-scale problems.

---

## Projects

### [linkedin-semantic-search](https://github.com/Vasan-th/linkedin-semantic-search)
Implementation of LinkedIn's production semantic search system ([arXiv:2602.07309](https://arxiv.org/abs/2602.07309)). Three-stage pipeline: dual-encoder EBR with InfoNCE + hard negatives → GPU RAR scoring → SLM reranking via multi-teacher distillation (8B oracle → 375M pruned model). Includes MixLM hybrid interaction (~76× throughput gain) and OSSCAR pruning.

### [computer-vision-suite](https://github.com/Vasan-th/computer-vision-suite)
CNN with residual blocks and Vision Transformer (ViT) from scratch on CIFAR-10. Four-model autoencoder progression on MNIST: Basic → Convolutional → Denoising → VAE (reparameterization trick). CLIP contrastive image-text training with learned temperature.


### [sequential-models-from-scratch](https://github.com/Vasan-th/sequential-models-from-scratch)
PyTorch implementations of RNN, LSTM, GRU, and Word2Vec (Skip-gram + CBOW) built from atomic equations to full training pipelines — each **numerically verified against `torch.nn`**. Demonstrates vanishing gradients empirically: RNN ≈ 50% vs LSTM ≈ 100% on a 150-step long-range dependency task.

---

## Areas

**Retrieval & Ranking** — dual encoders, dense retrieval, learning to rank, knowledge distillation
**Sequence Modeling** — RNNs, Transformers, attention mechanisms, state space models
**Generative Models** — VAEs, diffusion, CLIP-style contrastive learning
**Reinforcement Learning** — policy gradients, actor-critic, offline RL (CQL, IQL, AWAC)
**Graphs** — GCN, GraphSAGE, Node2Vec, heterogeneous graphs
**LLM Infra** — fine-tuning, RLHF/DPO/GRPO, inference optimization (vLLM, flash attention), DSPy

---

## Stack

Python · PyTorch · NumPy · scikit-learn · HuggingFace Transformers · LangChain · Spark

---
