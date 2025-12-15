import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
import numpy as np

# Early Stopping Class (Section IV.C)
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training Function (Section IV.C)
def train(model, train_loader, criterion, optimizer, num_epochs, scheduler=None):
    training_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        training_losses.append(avg_loss)
        if scheduler:
            scheduler.step()
    return training_losses

# Evaluation Function with Metrics and Plots (Section IV.D/E)
def evaluate_metrics(model, test_loader, criterion, class_names, plot=False):
    model.eval()
    all_labels = []
    all_preds = []
    running_val_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_val_loss = running_val_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc_score = roc_auc_score(all_labels, label_binarize(all_preds, classes=range(len(class_names))), 
                             average='weighted', multi_class='ovo')

    if plot:
        plot_confusion_matrix(all_labels, all_preds, class_names)
        plot_roc_curve(all_labels, all_preds, class_names)

    return avg_val_loss, accuracy, precision, recall, f1, auc_score

# Confusion Matrix Plot (Section IV.D)
def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# ROC Curve Plot (Section IV.D)
def plot_roc_curve(labels, preds, class_names):
    labels_bin = label_binarize(labels, classes=range(len(class_names)))
    preds_bin = label_binarize(preds, classes=range(len(class_names)))

    plt.figure(figsize=(10, 7))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], preds_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    # Dummy test (replace with actual usage in main script)
    from torch.utils.data import DataLoader
    from dataset import MRI_Dataset
    data_dir = "."
    train_paths, test_paths, train_labels, test_labels = load_data(data_dir)
    transform = None  # Define transform in main script
    test_dataset = MRI_Dataset(test_paths, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = CNN_Transformer(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    evaluate_metrics(model, test_loader, criterion, ['Control', 'PD', 'Prodromal'], plot=True)