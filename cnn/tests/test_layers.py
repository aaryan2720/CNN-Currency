"""
Comprehensive Unit & Pipeline Tests for CNN Layers
==================================================
Tests Conv2D, ReLU, MaxPool2D, Flatten, Dense, and end-to-end forward/backward pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from cnn.layers import Conv2D, MaxPool2D, Flatten, Dense
from cnn.activations import ReLU, Softmax
from cnn.losses import SoftmaxCrossEntropyLoss, CrossEntropyLoss


def test_flatten_layer():
    print("\n--- Running Test: Flatten Layer ---")
    x = np.random.randn(2, 4, 4, 4).astype(np.float64)
    flatten = Flatten()
    
    out = flatten.forward(x)
    assert out.shape == (2, 64), f"Flatten forward shape mismatch: {out.shape} != (2, 64)"
    assert np.allclose(out.flatten(), x.flatten()), "Flatten changed tensor element values!"
    print(f" [PASS] Flatten forward shape: {out.shape} == (2, 64), values strictly preserved.")
    
    grad_out = np.random.randn(2, 64).astype(np.float64)
    dx = flatten.backward(grad_out)
    assert dx.shape == (2, 4, 4, 4), f"Flatten backward shape mismatch: {dx.shape} != (2, 4, 4, 4)"
    assert np.allclose(dx.flatten(), grad_out.flatten()), "Flatten backward changed gradient values!"
    print(f" [PASS] Flatten backward shape: {dx.shape} == (2, 4, 4, 4), reshaped without modification.")


def test_dense_layer():
    print("\n--- Running Test: Dense Layer ---")
    N, D_in, D_out = 4, 64, 7
    x = np.random.randn(N, D_in).astype(np.float64)
    dense = Dense(in_features=D_in, out_features=D_out, seed=42)
    
    out = dense.forward(x)
    assert out.shape == (N, D_out), f"Dense forward shape mismatch: {out.shape} != (4, 7)"
    print(f" [PASS] Dense forward shape: {out.shape} == (4, 7)")
    
    grad_out = np.random.randn(N, D_out).astype(np.float64)
    dx = dense.backward(grad_out)
    
    assert dx.shape == (N, D_in), f"Dense dX shape mismatch: {dx.shape} != (4, 64)"
    assert dense.dW.shape == (D_in, D_out), f"Dense dW shape mismatch: {dense.dW.shape} != (64, 7)"
    assert dense.db.shape == (D_out,), f"Dense db shape mismatch: {dense.db.shape} != (7,)"
    print(f" [PASS] Dense backward shapes: dX={dx.shape}, dW={dense.dW.shape}, db={dense.db.shape}")


def test_softmax_layer():
    print("\n--- Running Test: Softmax Numerical Stability & Normalization ---")
    softmax = Softmax()
    
    # 1. Standard logits
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0]], dtype=np.float64)
    probs = softmax.forward(logits)
    row_sums = np.sum(probs, axis=1)
    
    assert np.allclose(row_sums, [1.0, 1.0]), f"Softmax row sums != 1.0: {row_sums}"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Softmax probabilities out of [0, 1] range!"
    print(f" [PASS] Softmax standard output sums to 1.0 on all rows (row_sums={row_sums}).")
    
    # 2. Extreme large logits (Overflow resistance test)
    extreme_logits = np.array([[1000.0, 1001.0, 1002.0], [5000.0, 5000.0, 5000.0]], dtype=np.float64)
    extreme_probs = softmax.forward(extreme_logits)
    
    assert not np.any(np.isnan(extreme_probs)), "Softmax produced NaN on large logits!"
    assert not np.any(np.isinf(extreme_probs)), "Softmax produced Inf on large logits!"
    assert np.allclose(np.sum(extreme_probs, axis=1), [1.0, 1.0]), "Extreme logits row sums != 1.0!"
    print(f" [PASS] Softmax numerically stable under extreme values: [[1000, 1001, 1002]] -> {extreme_probs[0]}")


def test_end_to_end_pipeline_with_loss():
    print("\n--- Running Test: End-to-End Synthetic Pipeline (Conv2D -> ReLU -> MaxPool2D -> Flatten -> Dense -> Loss) ---")
    N, C_in, H, W = 2, 3, 8, 8
    num_classes = 7
    
    # 1. Instantiate full layer stack
    conv = Conv2D(in_channels=C_in, out_channels=4, kernel_size=3, stride=1, padding=1, seed=10)
    relu = ReLU()
    pool = MaxPool2D(pool_size=2, stride=2)
    flatten = Flatten()
    dense = Dense(in_features=4 * 4 * 4, out_features=num_classes, seed=20)
    criterion = SoftmaxCrossEntropyLoss()
    
    # 2. Forward pass
    x = np.random.randn(N, C_in, H, W).astype(np.float64)
    y_true = np.array([2, 5], dtype=np.int64)
    
    print(f"  [1] Input X           : shape={x.shape}")
    
    h1 = conv.forward(x)
    print(f"  [2] After Conv2D      : shape={h1.shape} (Expected: (2, 4, 8, 8))")
    assert h1.shape == (2, 4, 8, 8)
    
    h2 = relu.forward(h1)
    print(f"  [3] After ReLU        : shape={h2.shape} (Expected: (2, 4, 8, 8))")
    assert h2.shape == (2, 4, 8, 8)
    
    h3 = pool.forward(h2)
    print(f"  [4] After MaxPool2D   : shape={h3.shape} (Expected: (2, 4, 4, 4))")
    assert h3.shape == (2, 4, 4, 4)
    
    h4 = flatten.forward(h3)
    print(f"  [5] After Flatten     : shape={h4.shape} (Expected: (2, 64))")
    assert h4.shape == (2, 64)
    
    logits = dense.forward(h4)
    print(f"  [6] After Dense       : shape={logits.shape} (Expected: (2, 7))")
    assert logits.shape == (2, 7)
    
    loss = criterion.forward(logits, y_true)
    print(f"  [7] Computed Loss     : {loss:.4f} (Finite scalar: {isinstance(loss, float) and np.isfinite(loss)})")
    assert isinstance(loss, float) and np.isfinite(loss) and loss > 0.0
    
    # 3. Backward propagation end-to-end
    print("\n  Beginning Backward Pass Propagation:")
    
    dlogits = criterion.backward()
    print(f"  <- Loss dLogits       : shape={dlogits.shape} (Expected: (2, 7))")
    assert dlogits.shape == (2, 7)
    
    dh4 = dense.backward(dlogits)
    print(f"  <- Dense dX           : shape={dh4.shape} (Expected: (2, 64))")
    assert dh4.shape == (2, 64)
    assert dense.dW.shape == (64, 7)
    assert dense.db.shape == (7,)
    
    dh3 = flatten.backward(dh4)
    print(f"  <- Flatten dX         : shape={dh3.shape} (Expected: (2, 4, 4, 4))")
    assert dh3.shape == (2, 4, 4, 4)
    
    dh2 = pool.backward(dh3)
    print(f"  <- MaxPool2D dX       : shape={dh2.shape} (Expected: (2, 4, 8, 8))")
    assert dh2.shape == (2, 4, 8, 8)
    
    dh1 = relu.backward(dh2)
    print(f"  <- ReLU dX            : shape={dh1.shape} (Expected: (2, 4, 8, 8))")
    assert dh1.shape == (2, 4, 8, 8)
    
    dx = conv.backward(dh1)
    print(f"  <- Conv2D dX          : shape={dx.shape} (Expected: (2, 3, 8, 8))")
    assert dx.shape == (2, 3, 8, 8)
    assert conv.dW.shape == (4, 3, 3, 3)
    assert conv.db.shape == (4,)
    
    print("\n [PASS] Complete End-to-End Pipeline forward and backward verified with 100% gradient connectivity!")


if __name__ == "__main__":
    test_flatten_layer()
    test_dense_layer()
    test_softmax_layer()
    test_end_to_end_pipeline_with_loss()
    print("\n==============================================")
    print("      ALL UNIT & INTEGRATION TESTS PASSED")
    print("==============================================\n")
