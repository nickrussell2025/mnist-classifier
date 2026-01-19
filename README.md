---
title: MNIST Digit Classifier
emoji: 🔢
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# MNIST Digit Classifier

ML Zoomcamp 2025 Capstone Project - Handwritten digit recognition using PyTorch.

## Hugging Face Deployment

Model can be tested on [Hugging Face spaces](https://huggingface.co/spaces/theBuggersMuddle/mnist-classifier)

## Project Overview

A neural network trained on the MNIST dataset to classify handwritten digits (0-9). The model is deployed using Gradio with a drawing interface where users can sketch digits for prediction.

## EDA
Please see mnist-eda.ipynb for details for EDA and model hyperparameter tuning

## Model Performance

- **Architecture**: Fully connected neural network with [256, 128, 64] hidden layers
- **Test Accuracy**: 98.39%
- **Dropout Rate**: 0.20
- **Framework**: PyTorch

## Dataset

- **Source**: MNIST handwritten digits
- **Training samples**: 60,000
- **Test samples**: 10,000
- **Image size**: 28×28 grayscale

## Technologies

- **Python**: 3.13.7
- **Package Manager**: uv
- **ML Framework**: PyTorch
- **UI Framework**: Gradio
- **Deployment**: Docker, HuggingFace Spaces

## Local Development
```bash
# Install dependencies
uv sync

# Train model
uv run python train.py

# Run inference interface
uv run python predict.py
```

## Docker
```bash
# Build image
docker build -t mnist-classifier .

# Run container
docker run -p 7860:7860 mnist-classifier
```

## Project Structure
```
├── train.py                # Model training script
├── predict.py              # Gradio inference interface
├── best_model_final.pth    # Trained model weights
├── pyproject.toml          # Python dependencies
├── uv.lock                 # Locked dependencies
├── Dockerfile              # Container configuration
└── README.md
```

## Preprocessing Pipeline

The model uses MNIST-standard preprocessing:
1. Crop to bounding box
2. Resize to fit 20×20 box (preserving aspect ratio)
3. Pad to 28×28 with 4-pixel border
4. Normalize with MNIST mean (0.1307) and std (0.3081)

## Links

- **Demo**: [Hugging Face](https://huggingface.co/spaces/theBuggersMuddle/mnist-classifier)
- **Course**: [ML Zoomcamp 2025](https://github.com/DataTalksClub/machine-learning-zoomcamp)