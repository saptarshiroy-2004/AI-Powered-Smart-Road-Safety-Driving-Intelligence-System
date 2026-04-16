import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class DriverDistractionDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path) 
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(data_dir="./data/processed/state-farm/train", batch_size=32, test_split=0.2):
    # Classes are c0 to c9
    classes = [f"c{i}" for i in range(10)]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    all_paths = []
    all_labels = []
    
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.exists(cls_dir):
            continue
        
        # Get all images in this class
        img_paths = glob.glob(os.path.join(cls_dir, "*.jpg"))
        all_paths.extend(img_paths)
        all_labels.extend([class_to_idx[cls_name]] * len(img_paths))
        
    if len(all_paths) == 0:
        raise ValueError(f"No images found in {data_dir}. Did you run preprocess_dataset.py?")
        
    # Split into Train and Validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=test_split, random_state=42, stratify=all_labels
    )
    
    # Transforms. Custom normalization for 1 channel grayscale.
    train_transforms = transforms.Compose([
        transforms.RandomRotation(10), # Apply minor rotation to prevent overfitting
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = DriverDistractionDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = DriverDistractionDataset(val_paths, val_labels, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")
    return train_loader, val_loader
