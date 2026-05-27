# Vietnamese Handwritten Text Recognition (DA-HTR)

A deep learning project focused on highly accurate Vietnamese Handwritten Text Recognition (HTR). This repository provides a modular, production-ready baseline using a CRNN (Convolutional Recurrent Neural Network) paired with CTC (Connectionist Temporal Classification) Loss, specifically tuned for the complexities of Vietnamese diacritics.

## Repository Structure

```text
vietnamese-da-htr/
├── src/
│   ├── data/
│   │   ├── dataset.py      # ViHTRDataset implementation and AspectGroupedBatchSampler
│   │   └── vocab.py        # Vocabulary loader and tokenizer logic
│   ├── models/
│   │   └── crnn.py         # ResNet34 + BiLSTM architecture definition
│   └── utils/
│       └── metrics.py      # CTC decoding, CER, WER, and D-CER evaluation metrics
├── checkpoints/            # Directory containing pretrained weights (e.g., nb01_crnn_best.pt)
├── test_samples/           # Directory containing sample images for inference
├── app.py                  # Interactive web interface using Gradio
├── infer.py                # Command-line script for single-image inference
├── train.py                # Main training loop with Mixed Precision and LR scheduling
└── requirements.txt        # Project dependencies
```

## Features

- **End-to-End Pipeline**: Directly maps raw images to Vietnamese text.
- **Dynamic Image Padding**: Employs Aspect-Ratio grouped batching to dynamically scale and pad images, drastically reducing memory overhead and empty convolutions.
- **Diacritic-Aware Evaluation**: Incorporates D-CER (Diacritic Character Error Rate), a custom metric defined to precisely measure the model's accuracy on Vietnamese tonal marks.
- **Interactive Web Demo**: Built-in application for real-time inference testing.

## Dataset

The model is trained and evaluated on the [Viet-Handwriting-OCR-v2](https://huggingface.co/datasets/5CD-AI/Viet-Handwriting-OCR-v2) dataset provided by 5CD-AI. 

### Dataset Structure
The dataset is released with two predefined splits:
- **Train**: 59,248 samples
- **Test**: 1,000 samples

---

## Technical Report

### 1. Architecture: CRNN + CTC
The model follows a standard but highly robust CRNN paradigm:
- **Feature Extractor**: A `ResNet34` backbone pretrained on ImageNet. It processes input images of height `128px` (with dynamic width) and outputs a dense 2D feature map.
- **Spatial Pooling**: An `AdaptiveAvgPool2d((1, None))` layer collapses the height dimension, transforming the 2D map into a 1D sequence of spatial features.
- **Sequence Modeling**: A 2-layer Bidirectional LSTM (`hidden_size=256`) processes the spatial sequence to model contextual dependencies between characters.
- **Transcription**: The CTC (Connectionist Temporal Classification) loss enables end-to-end training without frame-level alignment, mapping the sequence to character probabilities.

### 2. The Diacritic Challenge and D-CER Definition
Vietnamese relies heavily on diacritics (tone marks and vowel modifiers). Standard CER often obfuscates diacritic errors because missing a tone mark (e.g., `a` versus `á`) is treated identically to missing a base character, failing to reflect the true semantic degradation in Vietnamese text.

To address this, **this project introduces and defines D-CER (Diacritic Character Error Rate)** as a novel evaluation metric. D-CER isolates and applies a dedicated penalty to diacritic misclassifications, providing a much stricter and more accurate representation of the model's capability to comprehend Vietnamese tonal nuances.
- **Overall CER**: ~5.13%
- **D-CER**: ~9.72%

### 3. Optimization: Aspect-Ratio Batching
Training on variable-length text images traditionally involves zero-padding to the maximum width in a batch. To minimize wasted computation:
- The dataset computes the aspect ratio (Width / Height) of every image.
- The `AspectGroupedBatchSampler` clusters images with similar aspect ratios into the same batch.
- This ensures dynamic padding is kept to an absolute minimum, accelerating training by up to 30%.

---

## Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download Checkpoints
Ensure you have the pretrained weights (`nb01_crnn_best.pt`) and vocabulary (`vocab.json`) placed inside the `checkpoints/` directory.

### 3. Running the Web Demo
To launch the interactive UI:
```bash
python app.py
```
Open `http://localhost:7860` in your web browser.

### 4. CLI Inference
To test the model from the terminal:
```bash
python infer.py --ckpt checkpoints/nb01_crnn_best.pt --vocab checkpoints/vocab.json --img test_samples/sample_1.jpg
```

### 5. Training
To train the model from scratch on your own dataset:
```bash
python train.py --data_dir data/images/ --train_file data/train.txt --val_file data/val.txt --vocab data/vocab.json --epochs 50 --batch_size 32
```
