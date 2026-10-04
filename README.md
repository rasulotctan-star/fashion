# Fashion-MNIST CNN Classifier

A Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images from the **Fashion-MNIST** dataset into 10 different clothing categories.

## 📌 Project Overview

This project uses a CNN to recognize different types of clothing from 28×28 grayscale images.

The model:

* Loads the Fashion-MNIST dataset
* Normalizes pixel values from 0–255 to 0–1
* Reshapes the images to include a channel dimension
* Uses two convolutional layers
* Uses max pooling to reduce image dimensions
* Uses a fully connected layer for classification
* Uses Dropout to help reduce overfitting
* Trains for 10 epochs
* Saves the trained model as `fashion.keras`

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

## 📂 Project Structure

```text
project/
│
├── fashion.py
├── fashion.keras
└── README.md
```

## 📦 Installation

Make sure Python is installed, then install the required libraries:

```bash
pip install tensorflow numpy matplotlib
```

## 🚀 How to Run

Run the Python file:

```bash
python fashion.py
```

The first time you run the program, TensorFlow will download the Fashion-MNIST dataset automatically.

The model will train for **10 epochs** with a batch size of **64** and use 10% of the training data for validation.

After training, the model will be saved as:

```text
fashion.keras
```

The saving process is included at the end of the program.

## 🧠 Model Architecture

The CNN has the following structure:

```text
Input: 28 × 28 × 1

↓
Conv2D - 32 filters
↓
MaxPooling2D

↓
Conv2D - 64 filters
↓
MaxPooling2D

↓
Flatten

↓
Dense - 128 neurons
↓
Dropout - 30%

↓
Dense - 10 neurons
↓
Softmax
```

The final layer contains **10 neurons**, one for each Fashion-MNIST class.

## ⚙️ Training

The model uses:

```python
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]
```

These settings are defined in the `model.compile()` section.

Training is performed for:

* **10 epochs**
* **Batch size:** 64
* **Validation split:** 10%

## 👕 Fashion-MNIST Classes

Fashion-MNIST contains 10 clothing categories:

| Label | Class       |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

## 💾 Saved Model

After training, the trained neural network is saved as:

```text
fashion.keras
```

You can later load this model without retraining:

```python
import tensorflow as tf

model = tf.keras.models.load_model("fashion.keras")
```

## 🎯 Goal

The goal of this project is to demonstrate how a **Convolutional Neural Network** can be trained to recognize and classify images using the Fashion-MNIST dataset.

## 👨‍💻 Author
Rasul Ibrahimov
Created as a machine learning / deep learning project using Python and TensorFlow.
