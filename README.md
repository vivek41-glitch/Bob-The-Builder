# Bob The Builder

**IBM Bob as a Senior Deep Learning Engineer**

---

## Overview

Bob The Builder is an AI agent that uses IBM Bob to automatically optimize deep learning models for CIFAR-100 image classification. Bob acts like a senior deep learning engineer — it experiments with hyperparameters, architecture changes, and regularization strategies, automatically committing improvements with Git and rolling back failures.

---

## How It Works

1. Bob reads the training code (`train.py`)
2. Bob suggests a hyperparameter change
3. Training runs for 5–10 minutes
4. If accuracy improves → Bob commits with `git commit`
5. If accuracy drops → Bob rolls back with `git reset --hard`
6. Steps 2–5 repeat for 30 iterations
7. A Streamlit dashboard shows real-time progress

---

## Optimizations Bob Performs

| Category | Options Tested |
|---|---|
| Learning Rate | 0.0001, 0.0005, 0.001, 0.005, 0.01 |
| Batch Size | 32, 64, 128 |
| Optimizer | Adam, SGD, AdamW |
| Dropout Rate | 0.0, 0.2, 0.3, 0.5 |
| Batch Normalization | True, False |
| Weight Decay | 0.0, 1e-5, 1e-4, 1e-3 |
| Data Augmentation | True, False |

> **Total possible combinations:** 1,000+ &nbsp;|&nbsp; **Actual experiments run:** 30

---

## Results

| Metric | Value |
|---|---|
| Baseline Accuracy | 40.43% |
| Best Accuracy | 44.67% |
| Total Improvement | +4.24% |
| Relative Gain | +10.5% |
| Winning Change | Batch Normalization (+3.91%) |

### Key Discoveries

| Outcome | Change | Notes |
|---|---|---|
| ✅ Success | Batch Normalization | +3.91% improvement |
| ❌ Failed | Dropout | Made results worse |
| ❌ Failed | Weight Decay | No improvement |
| ❌ Failed | SGD Optimizer | Only 27.3% accuracy |
| ❌ Failed | Higher Learning Rate | Made results worse |
| ❌ Failed | Data Augmentation | No gain |

### Conclusion

Bob discovered that for CIFAR-100 with a CNN architecture, Batch Normalization alone provides the biggest improvement. Adding complexity — dropout, augmentation, different optimizers — makes results worse. **Simplicity wins.**

---

## Tech Stack

- Python 3.9+
- PyTorch
- Streamlit
- Git
- IBM Bob

---

## Project Structure

```
Bob-The-Builder/
├── train.py           # CNN model for CIFAR-100
├── optimizer.py       # Bob's control loop + Git automation
├── config.py          # Configuration (read-only)
├── objective.py       # Scoring logic for improvements
├── architecture.py    # Flexible CNN architecture
├── augmentation.py    # Data augmentation strategies
├── training.py        # Training loop with early stopping
├── dashboard.py       # Streamlit dashboard
├── requirements.txt   # Python dependencies
├── README.md
├── experiments.csv    # All 30 experiment logs
├── best_model.pth     # Best trained model (44.67%)
└── data/              # CIFAR-100 dataset
```

---

## Setup & Usage

```bash
# 1. Clone the repository
git clone https://github.com/vivek41-glitch/Bob-The-Builder.git
cd Bob-The-Builder

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run baseline (optional)
python train.py

# 4. Run Bob's optimization
python optimizer.py

# 5. Launch dashboard
streamlit run dashboard.py
```

---

## Expected Results

| Phase | Accuracy |
|---|---|
| Baseline | 40–45% |
| After Optimization | 44–50% |
| Improvement | +4–5% |

---

## Team — Team Destroyer

| Role | Name |
|---|---|
| Team Lead & ML Engineer | Vivek Dubey |
| Frontend & Dashboard | Areeba Abid |
| Model Architecture | Atikah Qaisar |
| Training Pipeline | Sam (SUDO) |
| Documentation & QA | Rashmika |

---

## Links

- **GitHub:** [github.com/vivek41-glitch/Bob-The-Builder](https://github.com/vivek41-glitch/Bob-The-Builder)

---

## License

MIT License — free to use, modify, and distribute.

---

*Bob The Builder — Can he build it? Yes he can.*
