import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import MRI_Dataset, load_data
from model import CNN_Transformer

# Set random seed for reproducibility (consistent with main.py)
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration (matches main.py and paper Section IV)
data_dir = '.'  # Update to your PPMI data directory
batch_size = 16
num_classes = 3
image_size = (224, 224)

# Data augmentation and normalization for testing (Section IV.B.2)
# Skipping random augmentations for test phase, keeping only resize and normalization
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.Grayscale(num_output_channels=3),  # Matches paper's RGB duplication
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load test data (80-20 split from paper, Section IV.A)
_, test_paths, _, test_labels = load_data(data_dir)
test_dataset = MRI_Dataset(test_paths, test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the trained model
model = CNN_Transformer(num_classes=num_classes)
model.load_state_dict(torch.load('cnn_transformer_pd.pth'))  # Assumes model saved from main.py
model.eval()  # Set to evaluation mode

# Evaluation function (your original code)
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovo')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

# Run evaluation
if __name__ == "__main__":
    print(f"Test samples: {len(test_dataset)}")
    evaluate_model(model, test_loader)