import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

# Add the project root to the path so we can import src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import src.config
from src.models.ocr_nn import OcrNet

def decode_prediction(logits):
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
    return decoded_preds[0] # Return the first one since we only infer one image

def infer(image_path, weight_path):
    # Load config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    net = OcrNet(src.config.common_config.num_class)
    net = net.to(device)
    
    # Load weights
    if os.path.exists(weight_path):
        try:
            net.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"Loaded weights from {weight_path}")
        except Exception as e:
            print(f"Failed to load weights: {e}")
            return
    else:
        print(f"Weight file not found: {weight_path}")
        return

    net.eval()

    # Load and preprocess image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return

    # Resize to expected size (144, 48)
    img_resized = cv2.resize(img, (144, 48))
    
    # Preprocess: normalize and permute
    # Input: 3, 48, 144
    image_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0) # Add batch dimension: [1, 3, 48, 144]
    image_tensor = image_tensor.to(device)

    # Inference
    with torch.no_grad():
        predict = net(image_tensor) # [Time, Batch, Class]
        predicted_text = decode_prediction(predict)

    print(f"Predicted Text: {predicted_text}")

    # Display result
    plt.figure(figsize=(10, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {predicted_text}")
    plt.axis('off')
    output_image_path = "inference_result.png"
    plt.savefig(output_image_path)
    print(f"Result image saved to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description="OCR Inference Script")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--weight_path", type=str, default="weights/ocr_net2.pth", help="Path to the model weights")
    
    args = parser.parse_args()
    
    infer(args.image_path, args.weight_path)

if __name__ == "__main__":
    main()
