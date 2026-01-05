import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import Trainer

if __name__ == "__main__":
    print("Starting training...")
    try:
        trainer = Trainer(load_parameters=True)
        trainer.train()
        print("Training completed successfully.")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
