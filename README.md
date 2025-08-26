# Distributional_Reinforcement_Learning

This repository implements a **Distributional Reinforcement Learning (DRL)** algorithm for
recommending optimal treatments for patients.

Unlike traditional RL (e.g., Q-learning) that estimates only the **expected return**, DRL
learns the **full return distribution** for each action. This richer signal captures
risk, variability, and tail behavior, enabling more informative clinical decisions.

---

## Key Features
- Distributional RL with return **distributions** over state–action pairs.
- Risk-aware decision support beyond expected value.
- Healthcare-focused examples for treatment recommendation.

---

## Repository Structure
- **`s11_distributional_reinforcement_learning.py`** – master script that computes the
  return distribution for all `(state, action)` pairs and runs the DRL workflow.

> Tip: keep experiment configs and outputs under `experiments/` (e.g., logs, seeds, plots).

---

## Getting Started
1. **Clone** the repo and enter the project directory.
2. (Optional) **Create a virtual env** and install your dependencies.
3. **Run the master script**:
   ```bash
   python s11_distributional_reinforcement_learning.py
