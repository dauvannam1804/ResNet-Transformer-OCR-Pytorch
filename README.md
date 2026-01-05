# ResNet Transformer OCR Pytorch

This repository is a fork of [ResNet-Transformer-OCR-Pytorch](https://github.com/PeterrrrLi/ResNet-Transformer-OCR-Pytorch). It implements an OCR model using a ResNet backbone and a Transformer encoder-decoder architecture.

## Setup

Ensure you have Python installed. Install the required dependencies:

```bash
pip install torch torchvision opencv-python pyyaml tqdm matplotlib numpy
```

## Data Preparation

The project expects the data to be organized in a specific structure. By default, the `config.yml` points to `data/dataset_lr_500`.

1.  **Directory Structure**:
    Create a root directory for your dataset (e.g., `data/dataset_lr_500`) and create three subdirectories: `train`, `val`, and `test`.

    ```
    data/dataset_lr_500/
    ├── train/
    │   ├── images
    │   │   ├── Scenario-A_Brazilian_track_00007_lr-004.png
    │   │   ├── Scenario-A_Brazilian_track_00008_lr-003.png
    │   │   ├── ...
    │   ├── labels.txt
    ├── val/
    │   ├── images
    │   │   ├── Scenario-A_Brazilian_track_00007_lr-004.png
    │   │   ├── Scenario-A_Brazilian_track_00008_lr-003.png
    │   │   ├── ...
    │   ├── labels.txt
    └── test/
    ```

2.  **Images and Labels**:
    *   Place your training images in `train/`, validation images in `val/`, and test images in `test/`.
    *   In each subdirectory, create a `labels.txt` file.
    *   **Format of `labels.txt`**: Each line should contain the relative image path (relative to the subdirectory) and the corresponding text label, separated by a space.

    Example `data/dataset_lr_500/train/labels.txt`:
    ```
    image_001.png HELLO
    image_002.jpg WORLD
    ...
    ```

3.  **Character Set**:
    Create a file (default: `data/chars.txt`) containing all possible characters that the model should recognize.
    
    Example `data/chars.txt`:
    ```
    0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
    ```

## Configuration

The training and model configuration is managed via `config.yml`. You can adjust parameters such as data paths, training epochs, batch size, and learning rate.

**`config.yml` Example:**

```yaml
common:
  data_root: "data/dataset_lr_500"  # Path to your dataset root
  weight: "weights/ocr_net2.pth"    # Path to save/load weights
  class_name_file: "data/chars.txt" # Path to character set file

train:
  epochs: 40
  batch_size: 64
  lr: 0.0001
  num_workers: 2
  save_dir: "weights"
```

## Training

To train the model, run the training script:

```bash
python scripts/run_train.py
```

*   The script will load configuration from `config.yml`.
*   It will save the best model (based on validation loss) and the model from the last epoch to the directory specified in `weight` (e.g., `weights/`).
*   Training logs are saved to `training_log.csv`.

## Inference

To run inference on a single image:

```bash
python scripts/infer.py --image_path path/to/your/image.png --weight_path weights/ocr_net2_best_epoch_XX.pth
```

*   `--image_path`: Path to the input image.
*   `--weight_path`: Path to the trained model weights (default: `weights/ocr_net2.pth`).

The script will print the predicted text and save a visualization to `inference_result.png`.
