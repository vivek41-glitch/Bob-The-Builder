import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from config import BASELINE_CONFIG, NUM_CLASSES, BEST_MODEL_FILE
from architecture import FlexibleCNN
from augmentation import get_train_transform, get_test_transform
from training import train_model
import os

def run_training(config=None):
    if config is None:
        config = BASELINE_CONFIG.copy()
    
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']
    optimizer_name = config['optimizer']
    dropout_rate = config['dropout_rate']
    use_batch_norm = config['use_batch_norm']
    weight_decay = config['weight_decay']
    augmentation = config['augmentation']
    patience = config['early_stopping_patience']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_transform = get_train_transform(augmentation)
    test_transform = get_test_transform()
    
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model = FlexibleCNN(use_batch_norm=use_batch_norm, dropout_rate=dropout_rate, num_classes=NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    best_accuracy, best_loss, training_time = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs, patience, device
    )
    
    return best_accuracy, best_loss, training_time

if __name__ == "__main__":
    acc, loss, t = run_training()
    print(f"\nFinal Results - Accuracy: {acc:.2f}%, Loss: {loss:.4f}, Time: {t:.2f}s")
    