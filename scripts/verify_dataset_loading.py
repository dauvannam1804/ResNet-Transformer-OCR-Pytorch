import sys
import os
# import torch
import cv2

# Add root directory to path so we can import utils
sys.path.append(os.getcwd())

from utils.dataset import OcrDataSet
import ocr_config

def verify_dataset():
    print("Initializing OcrDataSet(mode='train')...")
    try:
        dataset = OcrDataSet(mode='train')
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return

    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    print("Loading first sample...")
    try:
        image, target, target_length = dataset[0]
    except Exception as e:
        print(f"Failed to load sample: {e}")
        return

    print(f"Image shape: {image.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target length: {target_length}")
    
    # Decode label
    label_indices = target[:target_length].tolist()
    label_str = ""
    for idx in label_indices:
        if idx < len(ocr_config.class_name):
            label_str += ocr_config.class_name[idx]
        else:
            label_str += "?"
    
    print(f"Decoded Label: {label_str}")
    
    # Check if image tensor values are in [0, 1]
    print(f"Image min: {image.min()}, max: {image.max()}")

    if image.shape == (3, 48, 144):
        print("Image shape is correct.")
    else:
        print(f"Warning: Unexpected image shape {image.shape}")

if __name__ == "__main__":
    verify_dataset()
