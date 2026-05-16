import torchvision.transforms as transforms
from config import CIFAR100_MEAN, CIFAR100_STD

def get_train_transform(augmentation=False):
    if augmentation:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
        ])
    return transform

def get_test_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)
    ])
    return transform
    