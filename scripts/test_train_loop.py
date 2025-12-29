import sys
import os
import torch
from torch.utils.data import DataLoader, Subset

# Add root directory to path
sys.path.append(os.getcwd())

from src.train import Trainer
import src.config

def test_train_loop():
    print("Initializing Trainer...")
    trainer = Trainer(load_parameters=False)
    
    # Reduce dataset size for quick testing
    print("Reducing dataset size for testing...")
    trainer.train_dataset = Subset(trainer.train_dataset, range(10))
    trainer.val_dataset = Subset(trainer.val_dataset, range(5))
    
    trainer.train_loader = DataLoader(trainer.train_dataset, batch_size=2, shuffle=True)
    trainer.val_loader = DataLoader(trainer.val_dataset, batch_size=2, shuffle=False)
    
    print("Starting training loop for 1 epoch...")
    try:
        trainer.train(epochs=3)
        print("Training loop completed successfully.")
    except Exception as e:
        print(f"Training loop failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_train_loop()
