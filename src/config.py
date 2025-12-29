import os

import torch


class Config:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.weight = 'weights/ocr_net2.pth'
        
        # Load characters from file
        char_file = 'data/chars.txt'
        if os.path.exists(char_file):
            with open(char_file, 'r') as f:
                self.class_name = list(f.read().strip())
                # Add blank token at the beginning for CTC
                self.class_name.insert(0, '*') 
        else:
            # Fallback if file not found (should not happen if setup correctly)
            print(f"Warning: {char_file} not found. Using default alphanumeric.")
            self.class_name = ['*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.num_class = len(self.class_name)

# Create a global instance for backward compatibility or easy import
common_config = Config()

