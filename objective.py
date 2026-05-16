import time
from config import SCORING_WEIGHTS

def calculate_score(accuracy, loss, training_time):
    accuracy_score = accuracy * 100
    loss_score = max(0, (1.0 - loss) * 100)
    if loss_score > 100:
        loss_score = 100
    time_penalty = min(training_time / 60.0, 1.0)
    time_score = (1.0 - time_penalty) * 100
    final_score = (
        SCORING_WEIGHTS['accuracy_weight'] * accuracy_score +
        SCORING_WEIGHTS['loss_weight'] * loss_score +
        SCORING_WEIGHTS['time_penalty_weight'] * time_score
    )
    return round(final_score, 2)
    