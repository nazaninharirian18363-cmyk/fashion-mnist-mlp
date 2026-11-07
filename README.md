# Fashion-MNIST Classification with a Multilayer Perceptron (PyTorch)

A step-by-step implementation of a fully connected neural network (MLP) for classifying clothing images in the Fashion-MNIST dataset, built with PyTorch. The project covers data exploration, model design, training with regularization, evaluation, error analysis, and an ablation study.

## Results

| Model | Test Accuracy |
|---|---|
| Deep + Dropout (512-256, p=0.2) | **0.9016** |
| Wider (1024-512-256, p=0.3) | 0.9002 |
| Shallow (256) | 0.8977 |
| Deep (512-256), no dropout | 0.8967 |

The best model reaches about 90% test accuracy. Dropout improved the deeper network, and the gap between training and validation accuracy shows that the model still overfits somewhat.

## Project Structure
fashion-mnist-mlp/

├── notebooks/

│ └── fashion_mnist_mlp.ipynb # Full project notebook

├── results/

│ ├── class_distribution.png # Class balance in the training set

│ ├── samples.png # Example images from the dataset

│ ├── confusion_matrix.png # Confusion matrix on the test set

│ ├── misclassified.png # Examples the model got wrong

│ └── ablation.csv # Results of the four model configurations

├── requirements.txt


The trained model weights are saved to `models/` when the notebook runs. This folder is excluded from Git with `.gitignore`.

## Dataset

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) contains 70,000 grayscale images of 28×28 pixels in 10 clothing categories: 60,000 for training and 10,000 for testing. The notebook downloads it automatically with torchvision.

## Method

1. **Exploratory analysis:** check class balance and visualize sample images.
2. **Preprocessing:** convert images to tensors and normalize with the dataset mean (0.2860) and standard deviation (0.3530).
3. **Validation split:** hold out 10% of the training set for validation, so the test set is used only for final evaluation.
4. **Model:** an MLP with fully connected layers, ReLU activations, Batch Normalization, and Dropout.
5. **Training:** Adam optimizer (learning rate 1e-3, weight decay 1e-4), cross-entropy loss, early stopping (patience 5), and ReduceLROnPlateau (factor 0.5, patience 2).
6. **Evaluation:** test accuracy, classification report, and confusion matrix.
7. **Error analysis:** visualize misclassified images to see which classes get confused.
8. **Ablation study:** compare four configurations (depth, width, and dropout).
9. **Persistence:** save the trained model, reload it, and confirm the accuracy is unchanged.

## How to Run

The notebook runs in any environment with PyTorch installed, including Google Colab and GitHub Codespaces.

1. Open `notebooks/fashion_mnist_mlp.ipynb`.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Run all cells (**Runtime → Run all** in Colab).

A CPU is enough. Training takes a few minutes per configuration.

## Conclusion

- Deep networks with Dropout generalize better than the same network without it.
- Wider networks give a small gain, but not as much as Dropout.
- The reloaded model reached the same accuracy as the trained model, which confirms that saving and loading work.
- **Possible next steps:** a CNN, which suits image data better, data augmentation, and hyperparameter search with Optuna.

## Technologies

Python, PyTorch, torchvision, scikit-learn, NumPy, pandas, Matplotlib, Seaborn


├── .gitignore

└── README.md
