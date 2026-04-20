import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class HemorrhageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['hemorrhage']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def get_all_data(base_path="data/raw"):
    """Reads all original images recursively and prepares a base dataframe."""
    data = []
    categories = {"hemorrhage": 1, "normal": 0}
    
    for category, label in categories.items():
        cat_dir = os.path.join(base_path, category)
        if not os.path.exists(cat_dir):
            continue
            
        # Recursive search for all images
        for root, _, filenames in os.walk(cat_dir):
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, cat_dir)
                    # Use flattened relative path as orig_id to match augment_data.py naming
                    orig_id = rel_path.replace(os.sep, "_")
                    
                    data.append({
                        'id': f, 
                        'hemorrhage': label, 
                        'path': full_path,
                        'is_augmented': False,
                        'orig_id': orig_id
                    })
    return pd.DataFrame(data)

def get_raw_splits(base_path="data"):
    """Splits original raw data into train/val and test."""
    df_raw = get_all_data(os.path.join(base_path, "raw"))
    print(f"Loaded {len(df_raw)} original samples (including subfolders).")
    
    # Independent Test Set (15%) - Stratified
    train_val_df, test_df = train_test_split(
        df_raw, test_size=0.15, stratify=df_raw['hemorrhage'], random_state=42
    )
    return train_val_df, test_df

def add_augmented_data(train_df, base_path="data"):
    """Adds pre-generated augmented versions to the training dataframe."""
    aug_base = os.path.join(base_path, "augmented")
    if not os.path.exists(aug_base):
        return train_df
        
    aug_data = []
    categories = {"hemorrhage": 1, "normal": 0}
    
    # Get all original IDs in the train_df
    train_orig_ids = set(train_df['orig_id'].tolist())
    
    for category, label in categories.items():
        cat_dir = os.path.join(aug_base, category)
        if not os.path.exists(cat_dir): continue
        
        for f in os.listdir(cat_dir):
            # Filenames in augmented are like 'subfolder_name_aug_0.png'
            if "_aug_" in f:
                name_parts = f.split("_aug_")
                # The part before _aug_ is our clean orig_id but without extension
                # We need to match it back to orig_id (which has extension)
                aug_base_name = name_parts[0]
                ext = os.path.splitext(f)[1]
                
                # Check if aug_base_name + ext matches any orig_id in train
                match_id = aug_base_name + ext
                
                if match_id in train_orig_ids:
                    aug_data.append({
                        'id': f,
                        'hemorrhage': label,
                        'path': os.path.join(cat_dir, f),
                        'is_augmented': True,
                        'orig_id': match_id
                    })
    
    if aug_data:
        df_aug = pd.DataFrame(aug_data)
        print(f"Added {len(df_aug)} augmented samples to training set.")
        return pd.concat([train_df, df_aug], ignore_index=True)
    return train_df

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

def get_loaders(train_df, val_df, batch_size=32):
    # Add augmented images for the train set
    train_df = add_augmented_data(train_df)
    
    t_trans, v_trans = get_transforms()
    train_ds = HemorrhageDataset(train_df, t_trans)
    val_ds = HemorrhageDataset(val_df, v_trans)
    
    print(f"Final training set size: {len(train_df)} ({len(train_df[train_df['is_augmented']==False])} raw + {len(train_df[train_df['is_augmented']==True])} aug)")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Legacy support for app.py
def get_dataloaders(base_path="data", batch_size=32):
    train_val_df, test_df = get_raw_splits(base_path)
    train_df, val_df = train_test_split(train_val_df, test_size=0.15, stratify=train_val_df['hemorrhage'], random_state=42)
    t_loader, v_loader = get_loaders(train_df, val_df, batch_size)
    _, v_trans = get_transforms()
    test_ds = HemorrhageDataset(test_df, v_trans)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return t_loader, v_loader, test_loader, train_df, val_df, test_df
