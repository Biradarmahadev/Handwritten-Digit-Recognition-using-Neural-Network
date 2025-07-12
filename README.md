# Handwritten Digit Recognition using Neural Network

This project implements a neural network for recognizing handwritten digits using TensorFlow and Keras. The model is trained on a dataset of digit images and predicts the digit in unseen images.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [License](#license)

## Project Overview

The goal is to classify grayscale images of handwritten digits (0-9) using a neural network. The project uses a fully connected neural network (MLP) and demonstrates data preprocessing, model training, evaluation, and prediction.

## Dataset

- **Train.csv**: Contains labeled training data. The first column is the digit label (0-9), and the remaining columns are pixel values (flattened 28x28 images).
- **test.csv**: Contains unlabeled test data. Each row is a flattened 28x28 image.

## Installation

Run the following command in your Jupyter notebook to install dependencies:

```python
%pip install tensorflow pandas matplotlib scikit-learn
```

## Usage

Open `main.ipynb` and run the cells sequentially:

1. **Import Libraries**: Loads required Python packages.
2. **Load Data**: Reads training data from `Train.csv`.
3. **Preprocess Data**: Normalizes pixel values and reshapes data for the neural network.
4. **Encode Labels**: Converts digit labels to one-hot encoded vectors.
5. **Split Data**: Splits data into training and validation sets.
6. **Build Model**: Defines the neural network architecture.
7. **Train Model**: Fits the model to the training data.
8. **Evaluate Model**: Tests the model on validation data and plots accuracy.
9. **Predict on Test Data**: Loads `test.csv`, predicts digits, and visualizes results.

## Model Architecture

- **Input Layer**: Accepts 28x28 grayscale images.
- **Flatten Layer**: Converts 2D images to 1D vectors.
- **Dense Layer 1**: 128 neurons, ReLU activation.
- **Dense Layer 2**: 64 neurons, ReLU activation.
- **Output Layer**: 10 neurons, Softmax activation (for 10 digit classes).

## Training

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Epochs**: 10
- **Batch Size**: 32
- **Validation Split**: 20%

Training progress and accuracy are plotted for both training and validation sets.

## Evaluation

After training, the model is evaluated on the validation set. Validation accuracy is printed and plotted.

## Prediction

The model predicts digits for images in `test.csv`. The first five predictions are visualized using matplotlib.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for