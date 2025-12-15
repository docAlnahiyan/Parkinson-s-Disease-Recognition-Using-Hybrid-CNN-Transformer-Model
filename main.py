import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import MRI_Dataset, load_data
from model import CNN_Transformer
from train_eval import train, evaluate_metrics, EarlyStopping
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration (Section IV.C)
data_dir = '.'  
batch_size = 16
num_classes = 3
learning_rate = 0.0001
num_epochs = 100  
image_size = (224, 224)
patience = 5
class_names = ['Control', 'PD', 'Prodromal']

# Load data
train_paths, test_paths, train_labels, test_labels = load_data(data_dir)

# Data augmentation and normalization (Section IV.B.2)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=3),  # Matches paper's RGB duplication
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Datasets and Loaders
train_dataset = MRI_Dataset(train_paths, train_labels, transform=transform)
test_dataset = MRI_Dataset(test_paths, test_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer, Scheduler (Section IV.C)
model = CNN_Transformer(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
early_stopping = EarlyStopping(patience=patience)

# Training and Evaluation Loop
training_losses = []
validation_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, 1, scheduler)  # 1 epoch at a time
    training_losses.extend(train_loss)
    
    val_loss, accuracy, precision, recall, f1, auc_score = evaluate_metrics(model, test_loader, criterion, class_names, plot=False)
    validation_losses.append(val_loss)
    test_accuracies.append(accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered")
        break

# Final Evaluation (Section IV.D)
print(f"\nFinal Metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\nAUC-ROC: {auc_score:.4f}")

# Plot Training and Validation Loss (Figure 3)
plt.figure(figsize=(8, 6))
plt.plot(range(len(training_losses)), training_losses, label="Training Loss", color="blue")
plt.plot(range(len(validation_losses)), validation_losses, label="Validation Loss", color="green")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Plot Test Accuracy (Figure 4)
plt.figure(figsize=(8, 6))
plt.plot(range(len(test_accuracies)), test_accuracies, label="Test Accuracy", color="orange")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Test Accuracy Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Final Plots (Section IV.D)
evaluate_metrics(model, test_loader, criterion, class_names, plot=True)

# Save the model (optional, not in paper but useful)
torch.save(model.state_dict(), 'cnn_transformer_pd.pth')