# Adaptive Query-Specific Retrieval with Contextual Bandits

## Project Overview

This project explores how reinforcement learning (RL), particularly contextual bandits and deep RL, can be applied to optimize hybrid information retrieval systems dynamically. By adapting retrieval parameters (like top-k, similarity thresholds, and hybrid weights) to the nature of each query, we aim to maximize retrieval relevance while minimizing cost (e.g., token length).

We implement and evaluate several algorithms, including:
- **Epsilon-Greedy (non-contextual MAB)**
- **LinUCB (contextual bandit)**
- **PPO (Proximal Policy Optimization)**
- **A2C (Advantage Actor-Critic)**

The system is evaluated using SQuAD and Natural Questions (NQ) datasets.

---

## Key Features

- Hybrid retrieval (dense + sparse: SBERT + BM25)
- Contextual features for queries (e.g., embeddings, query length, token diversity)
- Continuous and discrete action modeling for retrieval parameters
- Reinforcement learning agents trained to balance relevance and efficiency

---

## File Descriptions

- `CMAB_SQuAD.ipynb`: Notebook for training and evaluating models on the SQuAD dataset.
- `CMAB_NQ.ipynb`: Notebook for training and evaluating models on the Natural Questions dataset.
- `RL Project Report Group 6_MAB for InfoRetrieval.pdf`: Full project report detailing methodology, experiments, and results.

---

## How to Run

1. Install dependencies:
   ```bash
   pip install sentence-transformers faiss-cpu scikit-learn

2. Run notebooks in order:
CMAB_SQuAD.ipynb to test on SQuAD.
CMAB_NQ.ipynb to test on NQ.

3. Optionally, modify hyperparameters for PPO/A2C models within the respective cells.

## Results Summary
PPO achieved the best trade-off between accuracy and token cost.
A2C delivered high accuracy but with high token usage.
Epsilon-Greedy was most efficient but less adaptive.
LinUCB added modest gains through context-aware decisions.
