import os

import cv2
import torch
from torch.utils.data import Dataset

import src.config


class OcrDataSet(Dataset):
    
    '''
    Input: 3, 48, 144
    
    Output: 27, Batch, Num of Classes
    '''

    def __init__(self, mode='train'):
        super(OcrDataSet, self).__init__()
        self.mode = mode
        self.dataset = []
        
        # Determine root directory based on mode
        if mode == 'train':
            self.root_dir = 'data/dataset_lr/train'
        elif mode == 'val':
            self.root_dir = 'data/dataset_lr/val'
        elif mode == 'test':
            self.root_dir = 'data/dataset_lr/test'
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")

        # Read labels.txt
        label_file = os.path.join(self.root_dir, 'labels.txt')
        if not os.path.exists(label_file):
             # Fallback for test set if it doesn't have labels or different structure, 
             # but assuming standard structure for now based on exploration
             print(f"Warning: {label_file} not found.")
        else:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        label = parts[1]
                        # images are relative to the root_dir (e.g. images/filename.png)
                        full_img_path = os.path.join(self.root_dir, img_path)
                        self.dataset.append((full_img_path, label))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img_path, label_str = self.dataset[item]
        
        # Load image
        plate = cv2.imread(img_path)
        if plate is None:
            # Handle missing image gracefully or raise error
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Resize to expected size (144, 48)
        plate = cv2.resize(plate, (144, 48))

        target = []
        for char in label_str:
            try:
                target.append(src.config.common_config.class_name.index(char))
            except ValueError:
                # Handle unknown characters if necessary
                print(f"Warning: Character '{char}' not in class_name.")
                target.append(0) # Use 0 or some unknown token index

        image = torch.from_numpy(plate).permute(2, 0, 1).float() / 255.0
        target_length = torch.tensor(len(target)).long()
        target = torch.tensor(target).reshape(-1).long()
        
        # Pad target to fixed length (e.g., 15)
        _target = torch.full(size=(15,), fill_value=0, dtype=torch.long)
        min_len = min(len(target), 15)
        _target[:min_len] = target[:min_len]

        return image, _target, target_length



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    data_ocr = OcrDataSet()
    img, target, tl = data_ocr[0]
    img = torch.permute(img, [1,2,0])
    print(img.shape, target.shape,tl)
    plt.imshow(img)
    plt.show()