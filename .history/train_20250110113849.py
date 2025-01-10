import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import timm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from PIL import Image

# ==========================
# 1. Configuration Settings
# ==========================

# Paths
DATA_DIR = '/content/images'  # Adjust path based on where the dataset is located in Colab
BEST_MODEL_PATH = 'best_efficientnet_b0.pth'
FINAL_MODEL_PATH = 'efficientnet_b0_final.pth'

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
VALID_SPLIT = 0.2
RANDOM_SEED = 42

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# 2. Data Preparation
# ==========================

# Define transformations for training and validation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                         std=[0.229, 0.224, 0.225]),   # ImageNet stds
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the dataset with ImageFolder
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)

# Calculate dataset sizes
total_size = len(full_dataset)
val_size = int(total_size * VALID_SPLIT)
train_size = total_size - val_size

# Split the dataset into training and validation
torch.manual_seed(RANDOM_SEED)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Update transforms for validation dataset
val_dataset.dataset.transform = val_transforms

# Handle class imbalance using WeightedRandomSampler
train_labels = [full_dataset.targets[idx] for idx in train_dataset.indices]
train_counts = Counter(train_labels)
print(f"Training class distribution: {train_counts}")

# Calculate weights for each class
class_weights = 1. / torch.tensor([train_counts[0], train_counts[1]], dtype=torch.float)
print(f"Class weights: {class_weights}")

# Assign weights to each sample
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==========================
# 3. Model Definition
# ==========================

# Initialize EfficientNet-B0
model = timm.create_model('efficientnet_b0', pretrained=True)

# Modify the classifier for binary classification
num_features = model.get_classifier().in_features
model.classifier = nn.Linear(num_features, 2)  # 2 classes: valid and invalid

# Move the model to the specified device
model = model.to(device)

# ==========================
# 4. Loss Function and Optimizer
# ==========================

# Define the loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ==========================
# 5. Training and Validation Functions
# ==========================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

# ==========================
# 6. Training Loop
# ==========================

best_val_acc = 0.0

# Lists to store metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print("-" * 10)
    
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Step the scheduler
    scheduler.step()
    
    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("Best model saved.\n")
    else:
        print()

# Save the final model after training
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"Final model saved as '{FINAL_MODEL_PATH}'")

# ==========================
# 7. Evaluation
# ==========================

# Load the best model
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model = model.to(device)
model.eval()

def get_predictions(model, dataloader, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

# Get predictions and labels
preds, labels = get_predictions(model, val_loader, device)

# Calculate metrics
acc = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average='binary', pos_label=1)
recall = recall_score(labels, preds, average='binary', pos_label=1)
f1 = f1_score(labels, preds, average='binary', pos_label=1)
cm = confusion_matrix(labels, preds)

print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Valid', 'Invalid'], 
            yticklabels=['Valid', 'Invalid'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
