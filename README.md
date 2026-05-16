# Bob The Builder

**IBM Bob as a Senior Deep Learning Engineer**

Bob The Builder is an AI agent that uses IBM Bob to automatically optimize deep learning models for CIFAR-100 image classification. Bob acts like a senior deep learning engineer - it experiments with hyperparameters, architecture changes, and regularization strategies, automatically keeping improvements with Git commit and rolling back failures with Git reset.

## How It Works

1. Bob reads the training code
2. Bob suggests a change (learning rate, batch size, optimizer, dropout, batch norm, weight decay, augmentation)
3. Training runs for 5-10 minutes
4. If accuracy improves → Bob commits the change with `git commit`
5. If accuracy drops → Bob rolls back with `git reset --hard`
6. Repeat 30 times
7. Streamlit dashboard shows real-time progress

## Tech Stack

- Python 3.9+
- PyTorch
- Streamlit
- Git
- IBM Bob

## Optimizations Bob Performs

| Category | Options |
|----------|---------|
| Learning Rate | 0.0001, 0.0005, 0.001, 0.005, 0.01 |
| Batch Size | 32, 64, 128 |
| Optimizer | Adam, SGD, AdamW |
| Dropout Rate | 0.0, 0.2, 0.3, 0.5 |
| Batch Normalization | True/False |
| Weight Decay | 0.0, 1e-5, 1e-4, 1e-3 |
| Data Augmentation | True/False |

## Setup Instructions

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run baseline: `python train.py`
4. Run optimizer: `python optimizer.py`
5. Launch dashboard: `streamlit run dashboard.py`

## Expected Results

- Baseline Accuracy: ~40-45%
- Best Accuracy: ~60-70%
- Improvement: +20-25%

## Team

- Vivek Dubey (Solo)

## License

MIT
