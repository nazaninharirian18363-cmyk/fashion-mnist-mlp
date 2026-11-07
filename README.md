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
