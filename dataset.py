import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Dataset class for loading MRI scans
class MRI_Dataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        image = nib.load(file_path).get_fdata()
        
        # Preprocessing: Extract middle slice (Section IV.B.1 of paper)
        middle_slice = image[:, :, image.shape[2] // 2]
        image = np.array(middle_slice, dtype=np.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data loading function
def load_data(data_dir):
    classes = ['control', 'pd', 'prodromal']  # Matches paper's 3 classes (Section IV.A)
    file_paths = []
    labels = []

    for label, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.nii'):
                file_paths.append(os.path.join(class_dir, file_name))
                labels.append(label)
    
    # 80-20 train-test split as per paper (Section IV.A)
    return train_test_split(file_paths, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Example usage for debugging
    data_dir = "."  # Update to your actual data directory
    train_paths, test_paths, train_labels, test_labels = load_data(data_dir)
    print(f"Train samples: {len(train_paths)}, Test samples: {len(test_paths)}")