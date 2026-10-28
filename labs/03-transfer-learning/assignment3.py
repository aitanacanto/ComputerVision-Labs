# Aitana Canto Rico

import timm
import torch

from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, AutoAugment, AutoAugmentPolicy, ToTensor, Normalize
from collections import defaultdict

criterion = torch.nn.CrossEntropyLoss()

def create_subset(dataset, num_per_class=256, num_class=10):
    # Task 1
    # Return a balanced dataset that is a subset of the given dataset.

    class_counts = defaultdict(int)
    select_indices = []

    for index, (_, label) in enumerate(dataset): ## Element by element
        if class_counts[label] < num_per_class:
            select_indices.append(index)
            class_counts[label] += 1

        # If we have enougth examples for each class
        enough = True
        for c in range(num_class):
            if class_counts[c] < num_per_class:
                enough = False
                break

        if enough:
            break


    subset = torch.utils.data.Subset(dataset, select_indices)
    return subset

# Task 2.1
# Specify the training preprocessing pipeline.

train_transform = Compose([
    RandomResizedCrop(84),
    RandomHorizontalFlip(),
    AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Task 2.2
# Specify the testing preprocessing pipeline.

test_transform = Compose([
    transforms.Resize(84),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def test(model, loader):
    model.eval()
    running_loss = 0.0
    processed = 0
    correct = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(loader)):

            # Task 3
            # Perform evaluation using the test set.
            logits = model(x) # Prediction
            loss = criterion(logits, y) # Loss calculation

            running_loss += loss.item() # Acumulation Loss
            processed += x.size(0) # Num imag processed until now
            preds = torch.argmax(logits, dim=1) # Class most likely for each image
            correct += (preds == y).sum().item() # Num of correct preditions


    print(f'test loss: {running_loss / (i + 1):.3f}')
    return correct / processed

def train(model, loader, num_epoch=20):
    model.train()

    # Task 4.1
    # Freeze all modules of the model except the classifier.
    # Initialise an optimiser.
    # Replace the line below with your code.
    model.requires_grad_(False)  # Freezing all the model
    model.classifier.requires_grad_(True)  # Activate optimiser
    
    for name, param in model.named_parameters():
        if "blocks.6" in name or "conv_head" in name:
            param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(),weight_decay=0.0001)
    
    for epoch_i in range(num_epoch):
        running_loss = 0.0
        for i, (x, y) in tqdm(enumerate(loader)):

            # Task 4.2
            # Perform training using the training set.
            optimizer.zero_grad() 
            logits = model(x) 
            loss = criterion(logits, y) # Cross entropy
            loss.backward() 
            optimizer.step()
            running_loss += loss.item()

        print(f'epoch {epoch_i} training loss: {running_loss / (i + 1):.3f}')

if __name__ == "__main__":

    # Task 1 & 2 test code
    cifar10train = create_subset(datasets.CIFAR10('CIFAR10', train=True, transform=train_transform, download=True))
    train_loader = torch.utils.data.DataLoader(cifar10train, batch_size=32, shuffle=True, num_workers=4)
    print(f'training set size: {len(cifar10train)}')
    cifar10test = create_subset(datasets.CIFAR10('CIFAR10', train=False, transform=test_transform, download=True))
    test_loader = torch.utils.data.DataLoader(cifar10test, batch_size=32, shuffle=False, num_workers=4)
    print(f'test set size: {len(cifar10test)}')


    # Task 3 test code
    model = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=10)
    print(f'Task 3: test accuracy: {100 * test(model, test_loader):2f}%')


    # Task 4 test code
    train(model, train_loader)
    print(f'test accuracy: {100 * test(model, test_loader):2f}%')
