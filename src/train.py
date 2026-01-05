import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config
from src.models.ocr_nn import OcrNet
from src.utils.dataset import OcrDataSet


class Trainer:
    def __init__(self, load_parameters=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = OcrNet(src.config.common_config.num_class)

        # Ensure weights directory exists
        os.makedirs(os.path.dirname(src.config.common_config.weight), exist_ok=True)

        if os.path.exists(src.config.common_config.weight) and load_parameters:
            try:
                self.net.load_state_dict(
                    torch.load(src.config.common_config.weight, map_location="cpu")
                )
                print("Loaded pretrained weights.")
            except Exception as e:
                print(f"Failed to load weights: {e}")
        else:
            print("Training from scratch.")

        self.net = self.net.to(self.device)

        # Datasets
        self.train_dataset = OcrDataSet(mode="train")
        self.val_dataset = OcrDataSet(mode="val")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=64, shuffle=True, num_workers=2
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=64, shuffle=False, num_workers=2
        )

        self.loss_func = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)

        self.best_acc = 0.0

    def decode_prediction(self, logits):
        # logits: [Time, Batch, Class] -> [Batch, Time, Class]
        logits = logits.permute(1, 0, 2)
        probs = logits.log_softmax(2)
        preds = torch.argmax(probs, dim=2).detach().cpu().numpy()

        decoded_preds = []
        for pred in preds:
            pred_str = ""
            last_char = -1
            for char_idx in pred:
                if char_idx != 0 and char_idx != last_char:  # 0 is blank
                    pred_str += src.config.common_config.class_name[char_idx]
                last_char = char_idx
            decoded_preds.append(pred_str)
        return decoded_preds

    def decode_target(self, targets, target_lengths):
        decoded_targets = []
        idx = 0
        for length in target_lengths:
            target_indices = targets[idx : idx + length].tolist()
            target_str = "".join(
                [src.config.common_config.class_name[i] for i in target_indices]
            )
            decoded_targets.append(target_str)
            idx += length
        return decoded_targets

    def calculate_char_accuracy(self, preds, targets):
        total_chars = 0
        correct_chars = 0
        for pred, target in zip(preds, targets):
            # Match by position
            min_len = min(len(pred), len(target))
            matches = 0
            for i in range(min_len):
                if pred[i] == target[i]:
                    matches += 1
            correct_chars += matches
            total_chars += len(target)
        return correct_chars, total_chars

    def calculate_full_match_accuracy(self, preds, targets):
        total_samples = len(targets)
        correct_matches = 0
        for pred, target in zip(preds, targets):
            if pred == target:
                correct_matches += 1
        return correct_matches, total_samples

    def validate(self):
        self.net.eval()
        val_loss = 0.0
        total_chars = 0
        correct_chars = 0
        total_samples = 0
        correct_matches = 0

        with torch.no_grad():
            for batch_idx, (images, targets, target_lengths) in enumerate(tqdm(
                self.val_loader, desc="Validating"
            )):
                images = images.to(self.device)

                # Prepare targets for CTCLoss (flattened)
                flat_targets = torch.tensor([], dtype=torch.long)
                for i, length in enumerate(target_lengths):
                    flat_targets = torch.cat((flat_targets, targets[i][:length]), dim=0)
                flat_targets = flat_targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                # Forward
                predict = self.net(images)  # [Time, Batch, Class]

                # Loss
                input_lengths = torch.full(
                    size=(predict.size(1),),
                    fill_value=predict.size(0),
                    dtype=torch.long,
                )
                loss = self.loss_func(
                    predict.log_softmax(2), flat_targets, input_lengths, target_lengths
                )
                val_loss += loss.item()

                # Accuracy
                decoded_preds = self.decode_prediction(predict)
                decoded_targets = self.decode_target(flat_targets, target_lengths.cpu())
                correct, total = self.calculate_char_accuracy(decoded_preds, decoded_targets)
                correct_chars += correct
                total_chars += total

                # Full Match Accuracy
                matches, samples = self.calculate_full_match_accuracy(decoded_preds, decoded_targets)
                correct_matches += matches
                total_samples += samples

                if batch_idx == 0:
                    print(f"\nValidation Sample - Pred: {decoded_preds[0]}, Target: {decoded_targets[0]}")


        avg_loss = val_loss / len(self.val_loader)
        avg_acc = correct_chars / total_chars if total_chars > 0 else 0.0
        avg_full_match_acc = correct_matches / total_samples if total_samples > 0 else 0.0
        return avg_loss, avg_acc, avg_full_match_acc

    def train(self, epochs=40):
        best_loss = float("inf")
        last_saved_path = None
        best_saved_path = None
        
        # Initialize log file
        log_file = "training_log.csv"
        with open(log_file, "w") as f:
            f.write("epoch,train_loss,train_acc,train_full_match_acc,val_loss,val_acc,val_full_match_acc\n")
        
        for epoch in range(epochs):
            self.net.train()
            train_loss = 0.0
            train_correct_chars = 0
            train_total_chars = 0
            train_correct_matches = 0
            train_total_samples = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, targets, target_lengths in pbar:
                images = images.to(self.device)
                
                # Prepare targets
                flat_targets = torch.tensor([], dtype=torch.long)
                for i, length in enumerate(target_lengths):
                    flat_targets = torch.cat((flat_targets, targets[i][:length]), dim=0)
                flat_targets = flat_targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                # Forward
                predict = self.net(images)
                input_lengths = torch.full(
                    size=(predict.size(1),),
                    fill_value=predict.size(0),
                    dtype=torch.long,
                )

                loss = self.loss_func(
                    predict.log_softmax(2), flat_targets, input_lengths, target_lengths
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                
                # Accuracy
                decoded_preds = self.decode_prediction(predict)
                decoded_targets = self.decode_target(flat_targets, target_lengths.cpu())
                correct, total = self.calculate_char_accuracy(decoded_preds, decoded_targets)
                train_correct_chars += correct
                train_total_chars += total

                # Full Match Accuracy
                matches, samples = self.calculate_full_match_accuracy(decoded_preds, decoded_targets)
                train_correct_matches += matches
                train_total_samples += samples

                pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = train_loss / len(self.train_loader)
            avg_train_acc = train_correct_chars / train_total_chars if train_total_chars > 0 else 0.0
            avg_train_full_match_acc = train_correct_matches / train_total_samples if train_total_samples > 0 else 0.0
            
            # Validation
            val_loss, val_acc, val_full_match_acc = self.validate()
            
            print(
                f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Train Full Match Acc: {avg_train_full_match_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Full Match Acc: {val_full_match_acc:.4f}"
            )
            
            # Log to CSV
            with open(log_file, "a") as f:
                f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_train_acc:.4f},{avg_train_full_match_acc:.4f},{val_loss:.4f},{val_acc:.4f},{val_full_match_acc:.4f}\n")
            
            # Save Last (Rotating)
            current_last_path = src.config.common_config.weight.replace(
                ".pth", f"_last_epoch_{epoch+1}.pth"
            )
            torch.save(self.net.state_dict(), current_last_path)
            
            if last_saved_path and os.path.exists(last_saved_path) and last_saved_path != current_last_path:
                try:
                    os.remove(last_saved_path)
                except OSError as e:
                    print(f"Error deleting old last weight: {e}")
            last_saved_path = current_last_path

            # Save Best (Rotating)
            if val_loss < best_loss:
                best_loss = val_loss
                current_best_path = src.config.common_config.weight.replace(
                    ".pth", f"_best_epoch_{epoch+1}.pth"
                )
                torch.save(self.net.state_dict(), current_best_path)
                print(f"New best model saved: {current_best_path} (Val Loss: {val_loss:.4f})")
                
                if best_saved_path and os.path.exists(best_saved_path) and best_saved_path != current_best_path:
                    try:
                        os.remove(best_saved_path)
                    except OSError as e:
                        print(f"Error deleting old best weight: {e}")
                best_saved_path = current_best_path


if __name__ == "__main__":
    # Create weights directory if not exists
    os.makedirs(os.path.dirname(src.config.common_config.weight), exist_ok=True)

    trainer = Trainer(load_parameters=True)
    trainer.train(epochs=40)
