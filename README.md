# Fashion MNIST MLP Model

This project trains a Multi-Layer Perceptron (MLP) on the Fashion MNIST dataset and exports it to ONNX format.

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Training and Export

To train the model and export it to ONNX format, simply run:
```bash
python train_and_export.py
```

This will:
1. Download the Fashion MNIST dataset
2. Train an MLP model for 10 epochs
3. Generate training plots (saved as 'training_results.png')
4. Export the model to 'fmnist_mlp.onnx'

## Model Architecture

The model uses a simple MLP architecture:
- Input layer: 784 neurons (28x28 flattened)
- Hidden layer 1: 512 neurons with ReLU activation
- Hidden layer 2: 256 neurons with ReLU activation
- Output layer: 10 neurons (one for each class)
- Dropout rate: 0.2
- Input shape: (1, 28, 28)
- Output shape: (10,) for 10 classes

## Training Parameters

- Batch size: 128
- Learning rate: 0.01
- Optimizer: SGD with momentum (0.9)
- Loss function: Cross Entropy Loss
- Number of epochs: 10

## Training Results

The training progress will be displayed during training, and a plot of the training loss and accuracy will be saved as 'training_results.png'. 