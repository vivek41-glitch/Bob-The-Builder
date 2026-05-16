# CONFIGURATION FILE - READ ONLY

CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.2673, 0.2564, 0.2762]

NUM_CLASSES = 100
INPUT_SIZE = 32

BASELINE_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 10,
    'optimizer': 'adam',
    'dropout_rate': 0.0,
    'use_batch_norm': False,
    'weight_decay': 0.0,
    'augmentation': False,
    'early_stopping_patience': 3,
    'gradient_clip': 0.0,
}

SCORING_WEIGHTS = {
    'accuracy_weight': 0.7,
    'loss_weight': 0.2,
    'time_penalty_weight': 0.1,
}

EXPERIMENT_LOG_FILE = 'experiments.csv'
BEST_MODEL_FILE = 'best_model.pth'
