# Training Methodology, Experiments & Results

## 1. Experimental Training Configuration

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| **Model Architecture** | 2-Stage Sequential CNN | Baseline feature extractor with 2 conv-pool blocks |
| **Optimizer** | Vanilla SGD | Standard gradient descent step $\theta \leftarrow \theta - \eta \nabla L$ |
| **Learning Rate ($\eta$)** | `0.001` | Conservative step size ensuring stable descent without momentum |
| **Mini-Batch Size** | `16` | Balanced batch size providing regular gradient updates (18 batches/epoch) |
| **Training Epochs** | `20` | Sufficient for convergence analysis on the 280-sample dataset |
| **Training Samples** | `280` | 40 images per denomination across 7 classes |
| **Validation Samples** | `70` | 10 images per denomination across 7 classes |
| **Random Seed** | `42` | Complete deterministic reproducibility across all runs |

---

## 2. Preliminary 14-Image Memorization Sanity Test

Before training on the full dataset, an **Overfit Sanity Test** was executed on a tiny 14-sample subset (2 images per class $\times$ 7 classes):

- **Objective**: Verify that the backpropagation implementation, weight updates, and loss gradients can drive training error to zero on a small dataset.
- **Success Criterion**: Training accuracy $\ge 95\%$ and training loss $< 0.1$.
- **Starting State**: Loss = `2.2174`, Accuracy = `14.29%` (2/14)
- **Final State (Epoch 80)**: Loss = **`0.0331`**, Accuracy = **`100.00%` (14/14)**
- **Outcome**: **PASSED**. The model successfully memorized all 14 images with 100% precision.

---

## 3. Full Dataset Training Execution History

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy | Avg Grad Norm | Param Norm | Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.9610 | 13.93% | 1.9505 | 18.57% | 5.1819 | 10.9702 | 5.4s |
| **2** | 1.9391 | 16.07% | 1.9465 | 15.71% | 4.6512 | 10.9705 | 4.8s |
| **3** | 1.9286 | 17.14% | 1.9405 | 14.29% | 4.4444 | 10.9709 | 4.8s |
| **4** | 1.9219 | 16.07% | 1.9310 | 17.14% | 4.6955 | 10.9713 | 4.7s |
| **5** | 1.9123 | 18.57% | 1.9230 | 17.14% | 4.7441 | 10.9718 | 4.8s |
| **6** | 1.9025 | 23.93% | 1.9160 | 21.43% | 4.8688 | 10.9724 | 4.7s |
| **7** | 1.8949 | 23.21% | 1.9124 | 21.43% | 4.7280 | 10.9730 | 4.8s |
| **8** | 1.8865 | 26.79% | 1.9029 | 31.43% | 4.5998 | 10.9736 | 4.8s |
| **9** | 1.8807 | 29.29% | 1.8982 | 21.43% | 4.9930 | 10.9743 | 5.1s |
| **10** | 1.8709 | 32.14% | 1.8902 | 27.14% | 5.1371 | 10.9750 | 5.6s |
| **11** | 1.8643 | 30.71% | 1.8867 | 30.00% | 5.0673 | 10.9758 | 5.5s |
| **12** | 1.8519 | 31.43% | 1.8809 | 35.71% | 4.4445 | 10.9765 | 5.3s |
| **13** | 1.8450 | 35.36% | 1.8771 | 37.14% | 4.8753 | 10.9773 | 5.3s |
| **14** | 1.8348 | 39.29% | 1.8718 | 30.00% | 5.1336 | 10.9782 | 5.2s |
| **15** | 1.8274 | 36.79% | 1.8680 | 30.00% | 5.2901 | 10.9790 | 5.2s |
| **16** | 1.8222 | 35.71% | 1.8600 | 40.00% | 5.8493 | 10.9799 | 5.2s |
| **17** | 1.8042 | 44.64% | 1.8503 | 40.00% | 5.1615 | 10.9809 | 5.4s |
| **18** | 1.7959 | 47.86% | 1.8439 | 37.14% | 4.9834 | 10.9818 | 5.2s |
| **19** | **1.7878** | **45.36%** | **1.8383** | **44.29% (Best)** | **5.0683** | **10.9827** | **5.6s** |
| **20** | 1.7797 | 44.29% | 1.8332 | 40.00% | 5.5769 | 10.9837 | 5.5s |

---

## 4. Final Evaluation & Per-Class Accuracy

* **Evaluation Partition**: 70 validation images (10 images per class across 7 denominations).
* **Overall Validation Accuracy**: **`44.29%`** (31 / 70 images correct).
* **Macro-Average Accuracy**: **`44.29%`**.
* **Validation Cross-Entropy Loss**: **`1.8383`**.

### Detailed Per-Class Breakdown:
| Class Name | Validation Samples | Correct | Incorrect | Class Accuracy | Primary Characteristic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Rs_10** | 10 | 0 | 10 | **0.00%** | Chocolate brown note |
| **Rs_20** | 10 | 6 | 4 | **60.00%** | Greenish-yellow note |
| **Rs_50** | 10 | 6 | 4 | **60.00%** | Fluorescent blue note |
| **Rs_100** | 10 | 0 | 10 | **0.00%** | Lavender note |
| **Rs_200** | 10 | 2 | 8 | **20.00%** | Bright-yellow note |
| **Rs_500** | 10 | 8 | 2 | **80.00%** | Stone-grey note |
| **Rs_2000** | 10 | 9 | 1 | **90.00%** | Magenta note |

---

## 5. Artifacts & Checkpoints

- **Best Model Weights**: [`checkpoints/best_model.npz`](../checkpoints/best_model.npz)
- **Metadata**: [`checkpoints/checkpoint_metadata.json`](../checkpoints/checkpoint_metadata.json)
- **Training Loss Curve**: [`reports/training_loss.png`](../reports/training_loss.png)
- **Accuracy Curve**: [`reports/accuracy.png`](../reports/accuracy.png)
- **Confusion Matrix**: [`reports/confusion_matrix.png`](../reports/confusion_matrix.png)
- **Prediction Grid**: [`reports/prediction_examples.png`](../reports/prediction_examples.png)
- **Error Analysis Log**: [`reports/error_analysis.md`](../reports/error_analysis.md)
