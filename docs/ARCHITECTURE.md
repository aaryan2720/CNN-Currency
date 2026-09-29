# CNN Architecture & Mathematical Specifications

## 1. Network Overview & Block Diagram

```text
Input: (N, 3, 64, 64)
  │
  ▼
[Conv2D Block 1]
  ├── Conv2D(3 → 8, kernel=3, stride=1, padding=1)  ──> (N, 8, 64, 64)
  ├── ReLU()                                        ──> (N, 8, 64, 64)
  └── MaxPool2D(pool=2, stride=2)                   ──> (N, 8, 32, 32)
  │
  ▼
[Conv2D Block 2]
  ├── Conv2D(8 → 16, kernel=3, stride=1, padding=1) ──> (N, 16, 32, 32)
  ├── ReLU()                                        ──> (N, 16, 32, 32)
  └── MaxPool2D(pool=2, stride=2)                   ──> (N, 16, 16, 16)
  │
  ▼
[Classifier Head]
  ├── Flatten()                                     ──> (N, 4096)
  ├── Dense(4096 → 32)                              ──> (N, 32)
  ├── ReLU()                                        ──> (N, 32)
  └── Dense(32 → 7)                                 ──> (N, 7) [Logits]
  │
  ▼
[Loss Function]
  └── SoftmaxCrossEntropyLoss(Logits, Targets)      ──> Scalar Loss
```

---

## 2. Layer Dimensionality & Parameter Table

| Layer # | Layer Type | Input Shape | Output Shape | Learnable Parameters | Formula / Configuration |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **Input** | — | $(N, 3, 64, 64)$ | 0 | Raw normalized RGB image |
| **1** | **Conv2D** | $(N, 3, 64, 64)$ | $(N, 8, 64, 64)$ | **224** | $W: (8, 3, 3, 3) = 216, b: (8,) = 8$ |
| **2** | **ReLU** | $(N, 8, 64, 64)$ | $(N, 8, 64, 64)$ | 0 | $\max(0, x)$ |
| **3** | **MaxPool2D** | $(N, 8, 64, 64)$ | $(N, 8, 32, 32)$ | 0 | Pool $2\times 2$, Stride 2 |
| **4** | **Conv2D** | $(N, 8, 32, 32)$ | $(N, 16, 32, 32)$ | **1,168** | $W: (16, 8, 3, 3) = 1152, b: (16,) = 16$ |
| **5** | **ReLU** | $(N, 16, 32, 32)$ | $(N, 16, 32, 32)$ | 0 | $\max(0, x)$ |
| **6** | **MaxPool2D** | $(N, 16, 32, 32)$ | $(N, 16, 16, 16)$ | 0 | Pool $2\times 2$, Stride 2 |
| **7** | **Flatten** | $(N, 16, 16, 16)$ | $(N, 4096)$ | 0 | $16 \times 16 \times 16 = 4096$ |
| **8** | **Dense** | $(N, 4096)$ | $(N, 32)$ | **131,104** | $W: (4096, 32) = 131072, b: (32,) = 32$ |
| **9** | **ReLU** | $(N, 32)$ | $(N, 32)$ | 0 | $\max(0, x)$ |
| **10** | **Dense** | $(N, 32)$ | $(N, 7)$ | **231** | $W: (32, 7) = 224, b: (7,) = 7$ |
| **TOTAL** | — | — | — | **132,727** | **132,727 Trainable Parameters** |

---

## 3. Mathematical Equations & Backpropagation

### A. 2D Convolution (`Conv2D`)
- **Forward Pass**:
  $$Z[n, f, i, j] = \sum_{c=0}^{C_{\text{in}}-1} \sum_{u=0}^{K_h-1} \sum_{v=0}^{K_w-1} X_{\text{pad}}[n, c, i \cdot S_h + u, j \cdot S_w + v] \cdot W[f, c, u, v] + b[f]$$
- **Weight Initialization (He Normal)**:
  $$\sigma = \sqrt{\frac{2}{C_{\text{in}} \cdot K_h \cdot K_w}}, \quad W \sim \mathcal{N}(0, \sigma^2), \quad b = 0$$
- **Backward Gradients**:
  $$db = \sum_{n, i, j} dZ[n, f, i, j] = \texttt{np.sum(dZ, axis=(0, 2, 3))}$$
  $$dW = \sum_{n, i, j} dZ[n, f, i, j] \cdot X_{\text{pad}}[n, c, i \cdot S_h + u, j \cdot S_w + v]$$
  $$dX_{\text{pad}}[n, c, i \cdot S_h + u, j \cdot S_w + v] += \sum_f dZ[n, f, i, j] \cdot W[f, c, u, v]$$

### B. 2D Max Pooling (`MaxPool2D`)
- **Forward Pass**:
  $$Y[n, c, i, j] = \max_{u, v} X[n, c, i \cdot S_h + u, j \cdot S_w + v]$$
- **Backward Pass**:
  $$dX[n, c, i \cdot S_h + u, j \cdot S_w + v] = dY[n, c, i, j] \cdot \frac{\mathbb{I}(X = Y)}{\sum \mathbb{I}(X = Y)}$$

### C. Fully Connected Layer (`Dense`)
- **Forward Pass**:
  $$Y = XW + b$$
- **Backward Pass**:
  $$dW = X^T @ dY, \quad db = \sum_{n=1}^N dY[n, :], \quad dX = dY @ W^T$$

### D. Softmax Cross-Entropy Loss (`SoftmaxCrossEntropyLoss`)
- **Forward Pass (Log-Sum-Exp Stabilization)**:
  $$L = -\frac{1}{N} \sum_{n=1}^N \left( Z_{n, y_n} - \max_k Z_{n, k} - \log \sum_j \exp(Z_{n, j} - \max_k Z_{n, k}) \right)$$
- **Backward Pass (Exact Analytical Cancellation)**:
  $$dZ = \frac{P - Y_{\text{one\_hot}}}{N}$$
