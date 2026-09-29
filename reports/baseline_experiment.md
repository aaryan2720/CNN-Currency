# Experiment Card: Baseline From-Scratch CNN (Experiment 1)
=========================================================

## 1. Overview
This experiment card provides the authoritative scientific record for the initial baseline Convolutional Neural Network (CNN) trained from scratch in pure Python and NumPy on a curated 350-image subset of Indian currency banknotes.

---

## 2. Dataset Specifications
* **Source**: Indian Currency Notes Dataset, Mendeley Data, Version 1 (`https://data.mendeley.com/datasets/48ympv8jjf/1`).
* **Licensing**: Please consult the original Mendeley Data page for current licensing and usage terms.
* **Curated Subset**: 350 total images (exactly 50 images per denomination across 7 classes).
* **Classes (7)**: `Rs_10`, `Rs_20`, `Rs_50`, `Rs_100`, `Rs_200`, `Rs_500`, `Rs_2000`.
* **Selection Rule**: First 50 valid images per class ordered lexicographically by source filename, deduplicated via SHA-256 (350 unique hashes, 0 duplicates).
* **Partitioning**: Stratified 80/20 train/validation split (Seed: `42`).
  - **Training Set ($N_{\text{train}}$)**: 280 images (exactly 40 images/class).
  - **Validation Set ($N_{\text{val}}$)**: 70 images (exactly 10 images/class).
  - **Data Leakage**: 0 overlapping files or SHA-256 hash collisions ($Train \cap Val = \emptyset$).
* **Input Tensor Shape**: $(N, 3, 64, 64)$ float32, normalized to $[0.0, 1.0]$.
* **Dataset Statistics**:
  - **Training Global Mean**: `0.4989` | **Training Global Std**: `0.1947`
  - **Validation Global Mean**: `0.4898` | **Validation Global Std**: `0.1946`
  - **Per-Channel Means (Train)**: Red=`0.5169`, Green=`0.4943`, Blue=`0.4855`
  - **Per-Channel Stds (Train)**: Red=`0.1886`, Green=`0.1947`, Blue=`0.1990`

---

## 3. Architecture & Parameters
$$\text{Input } (N, 3, 64, 64) \to \text{Conv2D}(3\to8) \to \text{ReLU} \to \text{MaxPool}(2,2) \to \text{Conv2D}(8\to16) \to \text{ReLU} \to \text{MaxPool}(2,2) \to \text{Flatten} \to \text{Dense}(4096\to32) \to \text{ReLU} \to \text{Dense}(32\to7)$$

| Layer # | Type | Input Shape | Output Shape | Trainable Parameters | Parameter Formula |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | `Conv2D` | $(N, 3, 64, 64)$ | $(N, 8, 64, 64)$ | **224** | $(8 \times 3 \times 3 \times 3) + 8$ |
| 2 | `ReLU` | $(N, 8, 64, 64)$ | $(N, 8, 64, 64)$ | **0** | — |
| 3 | `MaxPool2D` | $(N, 8, 64, 64)$ | $(N, 8, 32, 32)$ | **0** | Window $2\times 2$, Stride 2 |
| 4 | `Conv2D` | $(N, 8, 32, 32)$ | $(N, 16, 32, 32)$ | **1,168** | $(16 \times 8 \times 3 \times 3) + 16$ |
| 5 | `ReLU` | $(N, 16, 32, 32)$ | $(N, 16, 32, 32)$ | **0** | — |
| 6 | `MaxPool2D` | $(N, 16, 32, 32)$ | $(N, 16, 16, 16)$ | **0** | Window $2\times 2$, Stride 2 |
| 7 | `Flatten` | $(N, 16, 16, 16)$ | $(N, 4096)$ | **0** | $16 \times 16 \times 16 = 4096$ |
| 8 | `Dense` | $(N, 4096)$ | $(N, 32)$ | **131,104** | $(4096 \times 32) + 32$ |
| 9 | `ReLU` | $(N, 32)$ | $(N, 32)$ | **0** | — |
| 10 | `Dense` | $(N, 32)$ | $(N, 7)$ | **231** | $(32 \times 7) + 7$ |
| **Total** | — | — | — | **132,727** | **132,727 float64 parameters** |

---

## 4. Training & Optimization Configuration
* **Optimization Algorithm**: Vanilla Mini-Batch Stochastic Gradient Descent (SGD).
* **Update Equation**: $\theta \leftarrow \theta - \eta \nabla_{\theta} L$.
* **Learning Rate ($\eta$)**: `0.001` (constant, no schedule).
* **Batch Size**: `16` (18 batches per epoch: 17 batches of 16 + 1 batch of 8).
* **Epochs**: `20`.
* **Random Seed**: `42` (seeded Python, NumPy, and weight initialization).
* **Weight Initialization**: He (Kaiming) Normal ($\mathcal{N}(0, \sigma = \sqrt{2 / D_{\text{in}}})$), biases zero-initialized.
* **Loss Function**: Combined `SoftmaxCrossEntropyLoss` with log-sum-exp stabilization.
* **Regularization / Augmentation**: None (pure baseline).

---

## 5. Sanity Checks & Numerical Verification
* **Finite-Difference Gradient Checks (Float64)**:
  - Conv2D $W$: Relative Error = $8.38 \times 10^{-8}$ (PASS)
  - Conv2D $b$: Relative Error = $1.70 \times 10^{-8}$ (PASS)
  - Conv2D $X$: Relative Error = $8.09 \times 10^{-6}$ (PASS)
  - Dense $W$: Relative Error = $2.13 \times 10^{-8}$ (PASS)
  - Dense $b$: Relative Error = $4.55 \times 10^{-9}$ (PASS)
  - Dense $X$: Relative Error = $1.66 \times 10^{-8}$ (PASS)
  - SoftmaxCrossEntropy $Z$: Relative Error = $1.25 \times 10^{-7}$ (PASS)
* **14-Image Overfit Sanity Test (2 images/class × 7 classes)**:
  - Starting Loss: `2.2174` (Accuracy: 14.29%, 2/14)
  - Final Loss (Epoch 80): **`0.0331`** (Accuracy: **`100.00%`**, 14/14)
  - Memorization Criterion: Succeeded.

---

## 6. Authoritative Baseline Experimental Results

### Final Epoch Metrics (Epoch 20)
* **Training Loss**: `1.7797`
* **Training Accuracy**: `44.29%` (124 / 280 correct)
* **Validation Loss**: `1.8332`
* **Validation Accuracy**: `40.00%` (28 / 70 correct)
* **Average Gradient Norm**: `5.5769`
* **Parameter Norm**: `11.4013`

### Best Validation Checkpoint (Epoch 19)
* **Checkpoint File**: `checkpoints/best_model.npz`
* **Selection Criterion**: Highest validation accuracy (tie-breaker: lower validation loss).
* **Validation Accuracy**: **`44.29%` (31 / 70 correct)**
* **Validation Loss**: **`1.8383`**
* **Training Accuracy at Best Epoch**: `45.36%` (127 / 280 correct)
* **Training Loss at Best Epoch**: `1.7878`

---

## 7. Per-Class Validation Results (Best Checkpoint, $N_{\text{val}}=70$)

| Class | Denomination | Validation Samples | Correct | Incorrect | Accuracy (%) |
|---|---|:---:|:---:|:---:|:---:|
| `Rs_10` | 10 Rupee Note | 10 | 0 | 10 | **0.00%** |
| `Rs_20` | 20 Rupee Note | 10 | 6 | 4 | **60.00%** |
| `Rs_50` | 50 Rupee Note | 10 | 6 | 4 | **60.00%** |
| `Rs_100` | 100 Rupee Note | 10 | 0 | 10 | **0.00%** |
| `Rs_200` | 200 Rupee Note | 10 | 2 | 8 | **20.00%** |
| `Rs_500` | 500 Rupee Note | 10 | 8 | 2 | **80.00%** |
| `Rs_2000` | 2000 Rupee Note | 10 | 9 | 1 | **90.00%** |
| **Overall Summary** | — | **70** | **31** | **39** | **Overall: 44.29%** |

* **Macro-Average Accuracy**: $44.29\%$ (Note: Equal to overall accuracy because every class has an identical 10 validation images).

---

## 8. Scientific Limitations
1. **Limited Sample Size ($N=350$)**: With 40 training and 10 validation images per class, statistical uncertainty is $\pm 5.9\%$ (a single misclassification shifts class accuracy by 10%).
2. **Absence of Data Augmentation**: Without spatial shifts or mild brightness changes, the model is sensitive to capture backgrounds. Note: Horizontal/vertical flips should not be applied blindly because Indian banknotes have fixed asymmetric features (Gandhi portrait on the right, RBI seal, watermark window on the left).
3. **Low Spatial Resolution ($64 \times 64$)**: Downsampling eliminates security threads, micro-text, and numeral watermarks.
4. **Vanilla SGD vs Adaptive Optimizers**: Lack of momentum slows navigation through flat loss surfaces.

---

## 9. Exact Reproduction Command
```bash
python train.py --epochs 20 --batch-size 16 --lr 0.001 --seed 42
python evaluate.py
```
