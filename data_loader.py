import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class MedicalImageDataset(Dataset):
    def __init__(self, mri_paths, pet_paths):
        self.mri_paths = mri_paths
        self.pet_paths = pet_paths
        

        assert len(self.mri_paths) == len(self.pet_paths), "MRI和PET图像数量不匹配"
        
    def __len__(self):
        return len(self.mri_paths)

    def __getitem__(self, idx):
        mri_img = nib.load(self.mri_paths[idx]).get_fdata()
        ct_img = nib.load(self.pet_paths[idx]).get_fdata()

        if len(mri_img.shape) == 4:  

            mri_channels = mri_img.shape[3]
            mri_processed = np.zeros((mri_img.shape[0], mri_img.shape[1], mri_img.shape[2], mri_channels))
            
            for ch in range(mri_channels):
                mri_channel = mri_img[..., ch]
                if mri_channel.max() > mri_channel.min():
                    mri_processed[..., ch] = (mri_channel - mri_channel.min()) / (mri_channel.max() - mri_channel.min())
                else:
                    mri_processed[..., ch] = np.zeros_like(mri_channel)
        else:
            mri_channels = 1
            if mri_img.max() > mri_img.min():
                mri_processed = (mri_img - mri_img.min()) / (mri_img.max() - mri_img.min())
            else:
                mri_processed = np.zeros_like(mri_img)
            mri_processed = np.expand_dims(mri_processed, -1)  


        ct_min_pop = -1000.0
        ct_max_pop = 2000.0
        ct_img = np.clip(ct_img, ct_min_pop, ct_max_pop)
        ct_img = (ct_img - ct_min_pop) / (ct_max_pop - ct_min_pop)

        mri_img = torch.tensor(mri_processed, dtype=torch.float32).permute(3, 0, 1, 2)  # [channels, depth, height, width]
        ct_img = torch.tensor(ct_img, dtype=torch.float32).unsqueeze(0)  # [1, depth, height, width]
        
        filename = os.path.splitext(os.path.basename(self.pet_paths[idx]))[0]
        
        return {
            'MRI': mri_img, 
            'PET': ct_img,
            'MRI_path': self.mri_paths[idx], 
            'PET_path': self.pet_paths[idx],
            'filename': filename,
            'pet_min_orig': torch.tensor(ct_min_pop, dtype=torch.float32),
            'pet_max_orig': torch.tensor(ct_max_pop, dtype=torch.float32),
            'pet_min_pop': torch.tensor(ct_min_pop, dtype=torch.float32),
            'pet_p995_pop': torch.tensor(ct_max_pop, dtype=torch.float32)
        }

def get_train_val_test_loaders(
    data_dir, batch_size=4, val_size=0.1, test_size=0.2, 
    shuffle=True, num_workers=4, pin_memory=True
):
    
    mri_dir = os.path.join(data_dir, 'UTE')
    pet_dir = os.path.join(data_dir, 'CT')  
    
    if not os.path.exists(mri_dir):
        raise ValueError(f"MRI directory {mri_dir} is not found")
    
    if not os.path.exists(pet_dir):
        raise ValueError(f"PET directory {pet_dir} is not found")
    

    all_mri_paths = sorted([
        os.path.join(mri_dir, f) 
        for f in os.listdir(mri_dir) 
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])
    
    all_pet_paths = sorted([
        os.path.join(pet_dir, f) 
        for f in os.listdir(pet_dir) 
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ])
    
    # 检查数据完整性
    if len(all_mri_paths) == 0:
        raise ValueError(f"No MRI files found in directory {mri_dir}")
    
    if len(all_pet_paths) == 0:
        raise ValueError(f"No PET files found in directory {pet_dir}")
    
    if len(all_mri_paths) != len(all_pet_paths):
        raise ValueError(f"MRI file number ({len(all_mri_paths)}) does not match PET file number ({len(all_pet_paths)})")
    
    # 检查数据量是否足够划分
    total_samples = len(all_mri_paths)
    required_samples = 1 / (1 - val_size - test_size)
    
    if total_samples < required_samples:
        raise ValueError(
            f"Data is not enough, cannot divide the data into training set:validation set:test set in the ratio of {1-val_size-test_size:.0%}:{val_size:.0%}:{test_size:.0%} "
            f"The minimum number of samples required is {int(required_samples)}, but only {total_samples} samples are found"
        )
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_mri_paths, all_pet_paths, test_size=test_size, random_state=42, shuffle=True
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, shuffle=True
    )
    
    train_dataset = MedicalImageDataset(X_train, y_train)
    val_dataset = MedicalImageDataset(X_val, y_val)
    test_dataset = MedicalImageDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_paths = list(zip(X_test, y_test))
    
    print(f"Data loaded - training set: {len(train_dataset)} samples, validation set: {len(val_dataset)} samples, test set: {len(test_dataset)} samples")
    print(f"Division ratio: training set {len(train_dataset)/total_samples:.1%}, validation set {len(val_dataset)/total_samples:.1%}, test set {len(test_dataset)/total_samples:.1%}")
    
    return train_loader, val_loader, test_loader, test_paths