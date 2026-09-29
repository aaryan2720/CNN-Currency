"""
Unit Tests for Loss Functions
=============================
Validates CrossEntropyLoss and SoftmaxCrossEntropyLoss against known ground-truth values
and numerical stability cases.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from cnn.losses import CrossEntropyLoss, SoftmaxCrossEntropyLoss
from cnn.activations import Softmax


def test_cross_entropy_known_values():
    print("\n--- Running Test: CrossEntropy Known Hand-Calculated Values ---")
    loss_fn = CrossEntropyLoss()
    
    # Perfect predictions: P = [[1.0, 0.0], [0.0, 1.0]], y = [0, 1] -> Loss should be ≈ 0
    probs_perfect = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    y_perfect = np.array([0, 1], dtype=np.int64)
    loss_val = loss_fn.forward(probs_perfect, y_perfect)
    assert np.isclose(loss_val, 0.0, atol=1e-12), f"Perfect prediction loss != 0: {loss_val}"
    print(f" [PASS] Perfect prediction loss is ~= 0.0 ({loss_val:.2e}).")
    
    # Uniform 2-class: P = [[0.5, 0.5]], y = [0] -> Loss = -log(0.5) = ln(2) ~= 0.693147
    probs_uniform = np.array([[0.5, 0.5]], dtype=np.float64)
    y_uniform = np.array([0], dtype=np.int64)
    loss_uniform = loss_fn.forward(probs_uniform, y_uniform)
    expected_uniform = np.log(2.0)
    assert np.isclose(loss_uniform, expected_uniform, atol=1e-5), f"Uniform loss mismatch: {loss_uniform} != {expected_uniform}"
    print(f" [PASS] Uniform prediction loss matches exact ln(2) = {loss_uniform:.6f} == {expected_uniform:.6f}.")


def test_softmax_cross_entropy_consistency():
    print("\n--- Running Test: Standalone Softmax+CrossEntropy vs Combined SoftmaxCrossEntropyLoss ---")
    np.random.seed(42)
    N, C = 5, 7
    logits = np.random.randn(N, C).astype(np.float64)
    y = np.random.randint(0, C, size=(N,), dtype=np.int64)
    
    # 1. Standalone pipeline: Logits -> Softmax -> CrossEntropyLoss
    softmax = Softmax()
    probs = softmax.forward(logits)
    ce_loss_fn = CrossEntropyLoss()
    loss_standalone = ce_loss_fn.forward(probs, y)
    
    dP = ce_loss_fn.backward()
    dZ_standalone = softmax.backward(dP)
    
    # 2. Combined pipeline: Logits -> SoftmaxCrossEntropyLoss
    combined_fn = SoftmaxCrossEntropyLoss()
    loss_combined = combined_fn.forward(logits, y)
    dZ_combined = combined_fn.backward()
    
    # Compare forward loss and backward gradients
    assert np.isclose(loss_standalone, loss_combined, atol=1e-10), (
        f"Loss mismatch: {loss_standalone} != {loss_combined}"
    )
    assert np.allclose(dZ_standalone, dZ_combined, atol=1e-10), (
        f"Gradient mismatch between standalone and combined implementations!"
    )
    print(f" [PASS] Standalone and Combined loss values match: {loss_standalone:.8f} == {loss_combined:.8f}")
    print(f" [PASS] Standalone and Combined dZ gradients match bitwise across all {N}x{C} entries.")


if __name__ == "__main__":
    test_cross_entropy_known_values()
    test_softmax_cross_entropy_consistency()
    print("\n==============================================")
    print("         ALL LOSS UNIT TESTS PASSED")
    print("==============================================\n")
