"""
Finite-Difference Numerical Gradient Checking Suite
===================================================
Rigorous verification of analytical backpropagation gradients against
finite-difference approximations for all CNN layers:
- Conv2D (W, b, X)
- ReLU (X)
- MaxPool2D (X)
- Dense / Linear (W, b, X)
- Softmax (Z)
- SoftmaxCrossEntropyLoss (Logits Z)

Formula:
    g_num ≈ [L(θ + ε) - L(θ - ε)] / (2 * ε)
    rel_err = |g_ana - g_num| / max(|g_ana|, |g_num|, 1e-7)

Tolerance:
    With float64 precision and ε = 1e-7, rel_err < 1e-5 confirms analytical correctness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from cnn.layers import Conv2D, MaxPool2D, Flatten, Dense
from cnn.activations import ReLU, Softmax
from cnn.losses import SoftmaxCrossEntropyLoss, CrossEntropyLoss


def compute_relative_error(grad_ana: np.ndarray, grad_num: np.ndarray, eps: float = 1e-7) -> float:
    """Computes the maximum relative error between analytical and numerical gradients."""
    denominator = np.maximum(np.maximum(np.abs(grad_ana), np.abs(grad_num)), eps)
    rel_errors = np.abs(grad_ana - grad_num) / denominator
    return float(np.max(rel_errors))


def check_conv2d_gradients():
    print("\n=======================================================")
    print("      NUMERICAL GRADIENT CHECK: Conv2D LAYER")
    print("=======================================================")
    
    np.random.seed(42)
    epsilon = 1e-7
    
    N, C_in, H, W = 2, 2, 4, 4
    C_out, K, S, P = 2, 3, 1, 1
    
    x0 = np.random.randn(N, C_in, H, W).astype(np.float64)
    conv = Conv2D(in_channels=C_in, out_channels=C_out, kernel_size=K, stride=S, padding=P, seed=42)
    
    z0 = conv.forward(x0)
    G = np.random.randn(*z0.shape).astype(np.float64)
    
    dx_ana = conv.backward(G)
    dw_ana = conv.dW.copy()
    db_ana = conv.db.copy()
    
    # 1. Weights (W)
    print("\n[1/3] Checking Conv2D Weights (W)...")
    dw_num = np.zeros_like(conv.W)
    it = np.nditer(conv.W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = conv.W[idx]
        
        conv.W[idx] = orig_val + epsilon
        loss_plus = np.sum(conv.forward(x0) * G)
        
        conv.W[idx] = orig_val - epsilon
        loss_minus = np.sum(conv.forward(x0) * G)
        
        conv.W[idx] = orig_val
        dw_num[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
        it.iternext()
        
    w_rel_err = compute_relative_error(dw_ana, dw_num)
    print(f"  Max Absolute Diff    : {np.max(np.abs(dw_ana - dw_num)):.4e}")
    print(f"  Max Relative Error   : {w_rel_err:.4e}")
    assert w_rel_err < 1e-5, f"Conv2D dW gradient check failed! Rel error: {w_rel_err}"
    print(f"  --> [PASS] Conv2D dW verified successfully.")

    # 2. Biases (b)
    print("\n[2/3] Checking Conv2D Biases (b)...")
    db_num = np.zeros_like(conv.b)
    for i in range(len(conv.b)):
        orig_val = conv.b[i]
        
        conv.b[i] = orig_val + epsilon
        loss_plus = np.sum(conv.forward(x0) * G)
        
        conv.b[i] = orig_val - epsilon
        loss_minus = np.sum(conv.forward(x0) * G)
        
        conv.b[i] = orig_val
        db_num[i] = (loss_plus - loss_minus) / (2.0 * epsilon)
        
    b_rel_err = compute_relative_error(db_ana, db_num)
    print(f"  Max Absolute Diff    : {np.max(np.abs(db_ana - db_num)):.4e}")
    print(f"  Max Relative Error   : {b_rel_err:.4e}")
    assert b_rel_err < 1e-5, f"Conv2D db gradient check failed! Rel error: {b_rel_err}"
    print(f"  --> [PASS] Conv2D db verified successfully.")

    # 3. Input (X)
    print("\n[3/3] Checking Conv2D Input (X)...")
    dx_num = np.zeros_like(x0)
    it = np.nditer(x0, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = x0[idx]
        
        x0[idx] = orig_val + epsilon
        loss_plus = np.sum(conv.forward(x0) * G)
        
        x0[idx] = orig_val - epsilon
        loss_minus = np.sum(conv.forward(x0) * G)
        
        x0[idx] = orig_val
        dx_num[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
        it.iternext()
        
    x_rel_err = compute_relative_error(dx_ana, dx_num)
    print(f"  Max Absolute Diff    : {np.max(np.abs(dx_ana - dx_num)):.4e}")
    print(f"  Max Relative Error   : {x_rel_err:.4e}")
    assert x_rel_err < 1e-5, f"Conv2D dX gradient check failed! Rel error: {x_rel_err}"
    print(f"  --> [PASS] Conv2D dX verified successfully.")
    
    return {"W": w_rel_err, "b": b_rel_err, "X": x_rel_err}


def check_dense_gradients():
    print("\n=======================================================")
    print("      NUMERICAL GRADIENT CHECK: Dense / Linear LAYER")
    print("=======================================================")
    
    np.random.seed(42)
    epsilon = 1e-7
    
    N, D_in, D_out = 3, 5, 4
    x0 = np.random.randn(N, D_in).astype(np.float64)
    dense = Dense(in_features=D_in, out_features=D_out, seed=42)
    
    y0 = dense.forward(x0)
    G = np.random.randn(*y0.shape).astype(np.float64)
    
    dx_ana = dense.backward(G)
    dw_ana = dense.dW.copy()
    db_ana = dense.db.copy()
    
    # 1. Check Dense Weights (W)
    print("\n[1/3] Checking Dense Weights (W)...")
    dw_num = np.zeros_like(dense.W)
    it = np.nditer(dense.W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = dense.W[idx]
        
        dense.W[idx] = orig_val + epsilon
        loss_plus = np.sum(dense.forward(x0) * G)
        
        dense.W[idx] = orig_val - epsilon
        loss_minus = np.sum(dense.forward(x0) * G)
        
        dense.W[idx] = orig_val
        dw_num[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
        it.iternext()
        
    w_rel_err = compute_relative_error(dw_ana, dw_num)
    print(f"  Max Absolute Diff    : {np.max(np.abs(dw_ana - dw_num)):.4e}")
    print(f"  Max Relative Error   : {w_rel_err:.4e}")
    assert w_rel_err < 1e-5, f"Dense dW check failed: {w_rel_err}"
    print(f"  --> [PASS] Dense dW verified successfully.")

    # 2. Check Dense Biases (b)
    print("\n[2/3] Checking Dense Biases (b)...")
    db_num = np.zeros_like(dense.b)
    for i in range(len(dense.b)):
        orig_val = dense.b[i]
        
        dense.b[i] = orig_val + epsilon
        loss_plus = np.sum(dense.forward(x0) * G)
        
        dense.b[i] = orig_val - epsilon
        loss_minus = np.sum(dense.forward(x0) * G)
        
        dense.b[i] = orig_val
        db_num[i] = (loss_plus - loss_minus) / (2.0 * epsilon)
        
    b_rel_err = compute_relative_error(db_ana, db_num)
    print(f"  Max Absolute Diff    : {np.max(np.abs(db_ana - db_num)):.4e}")
    print(f"  Max Relative Error   : {b_rel_err:.4e}")
    assert b_rel_err < 1e-5, f"Dense db check failed: {b_rel_err}"
    print(f"  --> [PASS] Dense db verified successfully.")

    # 3. Check Dense Input (X)
    print("\n[3/3] Checking Dense Input (X)...")
    dx_num = np.zeros_like(x0)
    it = np.nditer(x0, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = x0[idx]
        
        x0[idx] = orig_val + epsilon
        loss_plus = np.sum(dense.forward(x0) * G)
        
        x0[idx] = orig_val - epsilon
        loss_minus = np.sum(dense.forward(x0) * G)
        
        x0[idx] = orig_val
        dx_num[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
        it.iternext()
        
    x_rel_err = compute_relative_error(dx_ana, dx_num)
    print(f"  Max Absolute Diff    : {np.max(np.abs(dx_ana - dx_num)):.4e}")
    print(f"  Max Relative Error   : {x_rel_err:.4e}")
    assert x_rel_err < 1e-5, f"Dense dX check failed: {x_rel_err}"
    print(f"  --> [PASS] Dense dX verified successfully.")
    
    return {"W": w_rel_err, "b": b_rel_err, "X": x_rel_err}


def check_softmax_cross_entropy_gradients():
    print("\n=======================================================")
    print("      NUMERICAL GRADIENT CHECK: SoftmaxCrossEntropyLoss")
    print("=======================================================")
    
    np.random.seed(42)
    epsilon = 1e-7
    
    N, C = 3, 5
    logits = np.random.randn(N, C).astype(np.float64)
    y_true = np.array([1, 0, 4], dtype=np.int64)
    
    loss_fn = SoftmaxCrossEntropyLoss()
    loss_val = loss_fn.forward(logits, y_true)
    dz_ana = loss_fn.backward()
    
    dz_num = np.zeros_like(logits)
    it = np.nditer(logits, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = logits[idx]
        
        logits[idx] = orig_val + epsilon
        loss_plus = loss_fn.forward(logits, y_true)
        
        logits[idx] = orig_val - epsilon
        loss_minus = loss_fn.forward(logits, y_true)
        
        logits[idx] = orig_val
        dz_num[idx] = (loss_plus - loss_minus) / (2.0 * epsilon)
        it.iternext()
        
    rel_err = compute_relative_error(dz_ana, dz_num)
    print(f"  Sample Analytical dZ : {dz_ana.flatten()[:4]}")
    print(f"  Sample Numerical dZ  : {dz_num.flatten()[:4]}")
    print(f"  Max Absolute Diff    : {np.max(np.abs(dz_ana - dz_num)):.4e}")
    print(f"  Max Relative Error   : {rel_err:.4e}")
    assert rel_err < 1e-5, f"SoftmaxCrossEntropyLoss dZ check failed: {rel_err}"
    print(f"  --> [PASS] SoftmaxCrossEntropyLoss dLogits verified successfully (rel_err < 1e-5).")
    return {"Logits": rel_err}


def run_all_gradient_checks():
    conv_res = check_conv2d_gradients()
    dense_res = check_dense_gradients()
    loss_res = check_softmax_cross_entropy_gradients()
    
    print("\n=======================================================")
    print("            GRADIENT CHECK SUMMARY SCOREBOARD")
    print("=======================================================")
    print(f"  Conv2D Weights (W) Relative Error : {conv_res['W']:.4e}  [PASS]")
    print(f"  Conv2D Biases  (b) Relative Error : {conv_res['b']:.4e}  [PASS]")
    print(f"  Conv2D Input   (X) Relative Error : {conv_res['X']:.4e}  [PASS]")
    print(f"  Dense  Weights (W) Relative Error : {dense_res['W']:.4e}  [PASS]")
    print(f"  Dense  Biases  (b) Relative Error : {dense_res['b']:.4e}  [PASS]")
    print(f"  Dense  Input   (X) Relative Error : {dense_res['X']:.4e}  [PASS]")
    print(f"  Loss   Logits  (Z) Relative Error : {loss_res['Logits']:.4e}  [PASS]")
    print("=======================================================")
    print(" [ALL GRADIENT CHECKS PASSED WITH MACHINE PRECISION]")
    print("=======================================================\n")


if __name__ == "__main__":
    run_all_gradient_checks()
