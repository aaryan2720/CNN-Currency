# From-Scratch Convolutional Neural Network for Indian Currency Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![NumPy](https://img.shields.io/badge/Implementation-Pure%20NumPy-orange.svg)
![Frameworks](https://img.shields.io/badge/Deep%20Learning%20Frameworks-None-green.svg)
![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Indian%20Currency-blueviolet.svg)
![Tests](https://img.shields.io/badge/Unit%20Tests%20%26%20GradChecks-Passing-success.svg)

An educational, academically rigorous implementation of a **2-Stage Convolutional Neural Network (CNN) built entirely from scratch in pure Python and NumPy**—without PyTorch, TensorFlow, Keras, torchvision, or scikit-learn.

The system is trained and evaluated on a certified 350-image subset across 7 Indian currency denominations (`Rs_10`, `Rs_20`, `Rs_50`, `Rs_100`, `Rs_200`, `Rs_500`, `Rs_2000`).

---

## 1. Project Overview & Motivation

Modern deep learning frameworks (e.g., PyTorch, TensorFlow) provide high-level abstractions like `nn.Conv2d` and automated differentiation (`autograd`). While efficient for production, these abstractions conceal the mathematical mechanics of convolution, tensor contraction, receptive field routing, and chain-rule backpropagation.

### Why Build a CNN from Scratch with NumPy?
- **Low-Level Mechanical Transparency**: Implements forward convolutions, spatial max pooling, and fully connected affine projections using explicit tensor calculus.
- **Analytical Gradient Derivation**: Manually derives and implements backward propagation gradients ($dX, dW, db$) for every layer.
- **Numerical Verification**: Validates analytical backpropagation gradients against finite-difference numerical approximations with floating-point machine precision.
- **End-to-End Control**: Implements streaming dataset extraction, deterministic stratified partitioning, mini-batch SGD, and evaluation metrics from first principles.

---

## 2. Technical Learning Objectives

1. Implement 2D convolution with configurable kernels, padding, and strides using vectorized tensor contraction.
2. Implement He (Kaiming) normal weight initialization to prevent activation collapse across deep ReLU networks.
3. Implement spatial max-pooling and route incoming backpropagation gradients exclusively to spatial argmax locations.
4. Implement numerically stable Softmax via max-shifting and log-sum-exp Cross-Entropy loss.
5. Derive the exact mathematical cancellation yielding $\frac{\partial L}{\partial Z} = \frac{P - Y}{N}$.
6. Verify gradient correctness using central finite differences: $\frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}$.
7. Train a complete mini-batch SGD pipeline and perform objective diagnostic evaluation.

---

## 3. Dataset: Original Archive vs. 350-Image Curated Subset

- **Original Dataset**: [Indian Currency Dataset - Mendeley Data (Version 1)](https://data.mendeley.com/datasets/48ympv8jjf/1)
- **Contributors**: Venkataramana Veeramsetty, Gaurav Singal, Tapas Badal
- **DOI**: `10.17632/48ympv8jjf.1`
- **Original Archive**: 11,657 images (4,657 camera captures + 7,000 augmented variations) totaling **10.65 GB**.

### Curated 350-Image Subset
For local CPU training, [`prepare_dataset.py`](prepare_dataset.py) streams a balanced subset of **350 images (exactly 50 per class)** directly via the Mendeley Data REST API:

```text
dataset/
├── Rs_10/    (50 images: image_001.jpg ... image_050.jpg)
├── Rs_20/    (50 images: image_001.jpg ... image_050.jpg)
├── Rs_50/    (50 images: image_001.jpg ... image_050.jpg)
├── Rs_100/   (50 images: image_001.jpg ... image_050.jpg)
├── Rs_200/   (50 images: image_001.jpg ... image_050.jpg)
├── Rs_500/   (50 images: image_001.jpg ... image_050.jpg)
└── Rs_2000/  (50 images: image_001.jpg ... image_050.jpg)
```

- **Deterministic Ordering**: Samples within each folder are sorted lexicographically by original filename to eliminate random selection bias.
- **Integrity Validation**: Every image is verified using Pillow (`img.verify()`, decode test, RGB conversion) and deduplicated via SHA-256 hashing.
- **Dataset Partitioning**: 80/20 Stratified Split $\to$ **280 Training Images** (40/class) and **70 Validation Images** (10/class) with **0% Data Leakage**.

*Detailed dataset documentation is available in [`docs/DATASET.md`](docs/DATASET.md).*

---

## 4. CNN Architecture & Parameter Summary

```text
Input: (N, 3, 64, 64)
  │
  ├── [Block 1] Conv2D(3 → 8, k=3, s=1, p=1) ──> ReLU ──> MaxPool2D(2, 2) ──> (N, 8, 32, 32)
  │
  ├── [Block 2] Conv2D(8 → 16, k=3, s=1, p=1) ─> ReLU ──> MaxPool2D(2, 2) ──> (N, 16, 16, 16)
  │
  └── [Classifier] Flatten ──> Dense(4096 → 32) ──> ReLU ──> Dense(32 → 7) ──> (N, 7) [Logits]
```

### Parameter Breakdown
| # | Layer Type | Input Shape | Output Shape | Trainable Parameters | Parameter Formula |
| :-: | :--- | :---: | :---: | :-: | :--- |
| 1 | **Conv2D** | $(N, 3, 64, 64)$ | $(N, 8, 64, 64)$ | **224** | $(8 \times 3 \times 3 \times 3) + 8$ |
| 2 | **ReLU** | $(N, 8, 64, 64)$ | $(N, 8, 64, 64)$ | **0** | — |
| 3 | **MaxPool2D** | $(N, 8, 64, 64)$ | $(N, 8, 32, 32)$ | **0** | Window $2\times 2$, Stride 2 |
| 4 | **Conv2D** | $(N, 8, 32, 32)$ | $(N, 16, 32, 32)$ | **1,168** | $(16 \times 8 \times 3 \times 3) + 16$ |
| 5 | **ReLU** | $(N, 16, 32, 32)$ | $(N, 16, 32, 32)$ | **0** | — |
| 6 | **MaxPool2D** | $(N, 16, 32, 32)$ | $(N, 16, 16, 16)$ | **0** | Window $2\times 2$, Stride 2 |
| 7 | **Flatten** | $(N, 16, 16, 16)$ | $(N, 4096)$ | **0** | $16 \times 16 \times 16 = 4096$ |
| 8 | **Dense** | $(N, 4096)$ | $(N, 32)$ | **131,104** | $(4096 \times 32) + 32$ |
| 9 | **ReLU** | $(N, 32)$ | $(N, 32)$ | **0** | — |
| 10 | **Dense** | $(N, 32)$ | $(N, 7)$ | **231** | $(32 \times 7) + 7$ |
| **TOTAL** | — | — | — | **132,727** | **132,727 Parameters** |

*Comprehensive mathematical layer derivations are provided in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).*

---

## 5. Mathematical Foundations & Backpropagation

### 1. Conv2D Forward & Backward
$$\text{Forward: } Z[n, f, i, j] = \sum_{c, u, v} X_{\text{pad}}[n, c, i \cdot S_h + u, j \cdot S_w + v] \cdot W[f, c, u, v] + b[f]$$
$$\text{Gradients: } db = \sum_{n, i, j} dZ, \quad dW = \sum_{n, i, j} dZ \cdot X_{\text{pad}}, \quad dX_{\text{pad}} = \sum_f dZ \cdot W$$

### 2. MaxPool2D Argmax Routing
$$dX[n, c, i \cdot S_h + u, j \cdot S_w + v] = dY[n, c, i, j] \cdot \frac{\mathbb{I}(X = Y)}{\sum \mathbb{I}(X = Y)}$$

### 3. Fully Connected (Dense) Projections
$$Y = XW + b \implies dW = X^T @ dY, \quad db = \sum_{n=1}^N dY[n, :], \quad dX = dY @ W^T$$

### 4. Softmax + Cross-Entropy Analytical Cancellation
$$L = -\frac{1}{N} \sum_{n=1}^N \log p_{n, y_n} \implies \frac{\partial L}{\partial Z} = \frac{P - Y_{\text{one\_hot}}}{N}$$

---

## 6. Finite-Difference Gradient Verification

To ensure that the analytical backpropagation implementation is mathematically exact, all layers were verified against numerical central finite differences in `float64` precision ($\epsilon = 10^{-7}$):

$$g_{\text{numerical}}(\theta) = \frac{L(\theta + \epsilon) - L(\theta - \epsilon)}{2\epsilon}, \quad \text{rel\_err} = \frac{|g_{\text{analytical}} - g_{\text{numerical}}|}{\max(|g_{\text{analytical}}|, |g_{\text{numerical}}|, 10^{-7})}$$

### Measured Verification Scoreboard:
| Component Checked | Tensor Shape | Max Absolute Difference | Max Relative Error | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Conv2D Weights ($W$)** | $(2, 2, 3, 3)$ | $1.2394 \times 10^{-8}$ | **$8.3781 \times 10^{-8}$** | **PASS** |
| **Conv2D Biases ($b$)** | $(2,)$ | $3.8641 \times 10^{-9}$ | **$1.6986 \times 10^{-8}$** | **PASS** |
| **Conv2D Input ($X$)** | $(2, 2, 4, 4)$ | $1.6685 \times 10^{-8}$ | **$8.0919 \times 10^{-6}$** | **PASS** |
| **Dense Weights ($W$)** | $(5, 4)$ | $3.7581 \times 10^{-9}$ | **$2.1314 \times 10^{-8}$** | **PASS** |
| **Dense Biases ($b$)** | $(4,)$ | $2.0517 \times 10^{-9}$ | **$4.5463 \times 10^{-9}$** | **PASS** |
| **Dense Input ($X$)** | $(3, 5)$ | $5.9176 \times 10^{-9}$ | **$1.6627 \times 10^{-8}$** | **PASS** |
| **SoftmaxCrossEntropy Logits ($Z$)** | $(3, 5)$ | $4.0962 \times 10^{-9}$ | **$1.2528 \times 10^{-7}$** | **PASS** |
| **ReLU Activations ($X$)** | $(2, 3)$ | $0.0000$ | **$3.0314 \times 10^{-9}$** | **PASS** |
| **MaxPool2D Activations ($X$)** | $(2, 2, 4, 4)$ | $0.0000$ | **$2.2401 \times 10^{-8}$** | **PASS** |

*Full gradient check analysis is documented in [`docs/GRADIENT_CHECKING.md`](docs/GRADIENT_CHECKING.md).*

---

## 7. Experimental Training Results

### A. 14-Image Memorization Sanity Check
Before full training, the network was tested on a 14-image subset (2 images per class):
- **Starting Loss**: `2.2174` $\to$ **Final Loss**: **`0.0331`**
- **Starting Accuracy**: `14.29%` $\to$ **Final Accuracy**: **`100.00%` (14/14 images)**

### B. 20-Epoch Full Training Metrics ($N_{\text{train}}=280, N_{\text{val}}=70$)
- **Batch Size**: `16` | **Learning Rate**: `0.001` (Vanilla SGD) | **Seed**: `42`
- **Best Validation Checkpoint (Epoch 19)**:
  - **Validation Accuracy**: **`44.29%` (31 / 70 images)**
  - **Validation Loss**: **`1.8383`**
  - **Training Loss**: `1.7878` | **Training Accuracy**: `45.36%`

| Metric | Training Loss Curve | Classification Accuracy Curve |
| :---: | :---: | :---: |
| **Curves** | ![Training Loss](reports/training_loss.png) | ![Accuracy](reports/accuracy.png) |

---

## 8. Diagnostic Evaluation & Error Analysis

### A. Per-Class Validation Performance ($N_{\text{val}}=70$)
| Class Name | Integer Label | Validation Samples | Correct | Incorrect | Class Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rs_10** | `0` | 10 | 0 | 10 | **0.00%** |
| **Rs_20** | `1` | 10 | 6 | 4 | **60.00%** |
| **Rs_50** | `2` | 10 | 6 | 4 | **60.00%** |
| **Rs_100** | `3` | 10 | 0 | 10 | **0.00%** |
| **Rs_200** | `4` | 10 | 2 | 8 | **20.00%** |
| **Rs_500** | `5` | 10 | 8 | 2 | **80.00%** |
| **Rs_2000** | `6` | 10 | 9 | 1 | **90.00%** |
| **Overall Summary** | — | **70** | **31** | **39** | **44.29%** |

### B. Confusion Matrix & Representative Predictions
| Validation Confusion Matrix | Sample Predictions (Correct & Misclassified) |
| :---: | :---: |
| ![Confusion Matrix](reports/confusion_matrix.png) | ![Prediction Examples](reports/prediction_examples.png) |

*Full sample-by-sample error analysis is documented in [`reports/error_analysis.md`](reports/error_analysis.md).*

---

## 9. Repository Structure

```text
CNN-currency/
├── cnn/
│   ├── __init__.py           # Package interface exposing SequentialCNN, layers, and SGD
│   ├── base.py               # Abstract Layer class
│   ├── layers.py             # Conv2D, MaxPool2D, Flatten, Dense implementations
│   ├── activations.py        # ReLU and Softmax implementations
│   ├── losses.py             # CrossEntropyLoss and SoftmaxCrossEntropyLoss
│   ├── optimizers.py         # Vanilla SGD optimizer
│   ├── model.py              # SequentialCNN container & checkpointing
│   └── tests/
│       ├── __init__.py
│       ├── test_layers.py    # Layer unit tests and integration pipeline
│       ├── test_losses.py    # Loss function unit tests
│       ├── test_model.py     # Architecture summary & 14-image overfit test
│       └── gradient_check.py # Finite-difference numerical gradient checks
│
├── dataset/                  # Curated 350-image dataset (50 images / class)
├── checkpoints/
│   ├── best_model.npz        # Serialized best model weights and biases
│   └── checkpoint_metadata.json # Checkpoint metadata and hyperparameter log
├── reports/
│   ├── training_loss.png     # Loss vs. epoch plot
│   ├── accuracy.png          # Accuracy vs. epoch plot
│   ├── confusion_matrix.png  # Confusion matrix heatmap
│   ├── confusion_matrix.csv  # Confusion matrix CSV data
│   ├── prediction_examples.png # Representative prediction sample grid
│   └── error_analysis.md     # In-depth validation error analysis
│
├── docs/
│   ├── DATASET.md            # Dataset extraction and validation documentation
│   ├── ARCHITECTURE.md       # Architectural specifications and mathematical derivations
│   ├── TRAINING.md           # Training methodology and experiment results
│   └── GRADIENT_CHECKING.md  # Numerical gradient checking documentation
│
├── prepare_dataset.py        # REST API dataset download and extraction script
├── dataset_loader.py         # Stratified pure NumPy dataset loader
├── dataset_metadata.json     # Complete dataset image index
├── validate_dataset.py       # Automated 7-point dataset validation suite
├── visual_inspection.py      # Dataset visual contact sheet generator
├── train.py                  # Full mini-batch training script
├── evaluate.py               # Checkpoint evaluation and error analysis script
├── training_history.json     # Machine-readable training logs
├── training_results.md       # Formatted experiment summary table
├── requirements.txt          # Minimal external dependencies
└── README.md                 # Master project documentation
```

---

## 10. Installation & Quickstart

### Prerequisites
- Python 3.10+
- Dependencies: `numpy`, `pillow`, `matplotlib`, `requests`, `urllib3`

```bash
# Clone the repository
git clone https://github.com/<your-username>/CNN-currency.git
cd CNN-currency

# Install minimal dependencies
pip install -r requirements.txt
```

### Reproducible Execution Commands
```bash
# 1. Stream & prepare dataset (if dataset directory is missing):
python prepare_dataset.py

# 2. Run automated dataset validation suite:
python validate_dataset.py

# 3. Run all unit and integration layer tests:
python cnn/tests/test_layers.py

# 4. Run finite-difference numerical gradient checking suite:
python cnn/tests/gradient_check.py

# 5. Run architecture check and 14-image overfit sanity test:
python cnn/tests/test_model.py

# 6. Train the CNN model on the full dataset:
python train.py --epochs 20 --batch-size 16 --lr 0.001 --seed 42

# 7. Evaluate the best checkpoint and generate diagnostic reports:
python evaluate.py --checkpoint checkpoints/best_model.npz
```

---

## 11. Scientific Limitations & Future Improvements

### Limitations of Current Baseline
1. **Dataset Scale**: The baseline training set consists of only **280 images (40 per class)** without data augmentation, leading to higher generalization variance on uncentered validation captures.
2. **Optimizer Simplicity**: Vanilla SGD without momentum or adaptive learning rates traverses loss landscapes slower than Adam or RMSprop.
3. **No Batch Normalization / Dropout**: Internal covariate shift and co-adaptation are unmitigated in the dense layers.
4. **Validation Sample Size**: The validation set contains **70 images**, meaning each sample represents a $1.43\%$ shift in overall accuracy.

### Future Research Directions
- Implement online NumPy data augmentations (random spatial translations, subtle rotation $\pm 10^\circ$, brightness jitter).
- Implement Adam / Momentum SGD optimizers and learning-rate decay schedulers.
- Implement Batch Normalization and Dropout layers in pure NumPy.
- Scale training to 1,000+ images per denomination using the full Mendeley repository.

---

## 12. Dataset Citation & Attribution

If you use this dataset or code for academic research, please cite the original authors:

```bibtex
@article{veeramsetty2020indian,
  title={Indian Currency Dataset},
  author={Veeramsetty, Venkataramana and Singal, Gaurav and Badal, Tapas},
  journal={Mendeley Data},
  volume={1},
  year={2020},
  doi={10.17632/48ympv8jjf.1},
  url={https://data.mendeley.com/datasets/48ympv8jjf/1}
}
```

*Licensing Terms: Creative Commons Attribution 4.0 International ([CC BY 4.0](http://creativecommons.org/licenses/by/4.0)).*
