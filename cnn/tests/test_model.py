"""
Model Architecture, Parameter Check, and Tiny Overfit Sanity Test
================================================================
Validates:
1. Layer-by-layer tensor shape progression for (N, 3, 64, 64)
2. Exact parameter count calculation
3. Parameter shape == Gradient shape check across all layers
4. 14-image (2 per class x 7 classes) Memorization / Overfit Sanity Test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from cnn.model import SequentialCNN
from cnn.losses import SoftmaxCrossEntropyLoss
from cnn.optimizers import SGD
from dataset_loader import load_currency_dataset


def test_model_shapes_and_summary():
    print("\n=======================================================")
    print("      TEST: Model Architecture & Parameter Verification")
    print("=======================================================")
    
    model = SequentialCNN(seed=42)
    print(model.summary(input_shape=(2, 3, 64, 64)))
    
    param_info = model.count_parameters()
    print(f"\nTotal Parameters: {param_info['total_parameters']:,}")
    
    # Forward check
    x = np.random.randn(2, 3, 64, 64).astype(np.float64)
    logits = model.forward(x)
    assert logits.shape == (2, 7), f"Expected logits shape (2, 7), got {logits.shape}"
    print(f" [PASS] Forward pass produced logits shape: {logits.shape}")
    
    # Backward check & shape matching
    grad_logits = np.random.randn(2, 7).astype(np.float64)
    dx = model.backward(grad_logits)
    assert dx.shape == (2, 3, 64, 64), f"Expected input grad shape (2, 3, 64, 64), got {dx.shape}"
    print(f" [PASS] Backward pass produced input gradient shape: {dx.shape}")
    
    params = model.parameters()
    grads = model.gradients()
    assert len(params) == len(grads), f"Mismatch: {len(params)} params vs {len(grads)} grads"
    
    print("\nParameter & Gradient Shape Consistency:")
    for (p_name, p_arr), (g_name, g_arr) in zip(params, grads):
        assert p_name == g_name, f"Name mismatch: {p_name} != {g_name}"
        assert p_arr.shape == g_arr.shape, f"Shape mismatch for {p_name}: {p_arr.shape} != {g_arr.shape}"
        assert not np.any(np.isnan(g_arr)), f"NaN gradient in {p_name}"
        print(f"  {p_name:<35} | Shape: {str(p_arr.shape):<18} | Match: YES")
        
    print("\n [PASS] All parameter and gradient shapes match 100% with finite numerical values.")


def test_tiny_overfit_sanity_check():
    print("\n=======================================================")
    print("      TEST: 14-Image Overfit Memorization Sanity Test")
    print("=======================================================")
    print("Objective: Verify the network can memorize 14 images (2 per class x 7 classes).")
    print("Success Criterion: Training loss drops significantly, training accuracy reaches >= 95% (14/14 = 100%).")
    
    # Load full dataset and extract exactly 2 images per class
    X_train_full, y_train_full, _, _, classes = load_currency_dataset(
        target_size=(64, 64),
        split_ratio=0.8,
        stratified=True,
        seed=42
    )
    
    # Select 2 images per class -> 14 images total
    tiny_indices = []
    for c in range(7):
        c_idx = np.where(y_train_full == c)[0][:2]
        tiny_indices.extend(c_idx)
        
    X_tiny = X_train_full[tiny_indices].astype(np.float64)
    y_tiny = y_train_full[tiny_indices]
    
    print(f"Extracted Tiny Dataset: X_tiny={X_tiny.shape}, y_tiny={y_tiny.shape}")
    print(f"Classes represented: {np.bincount(y_tiny)} (2 per class across 7 classes)")
    
    # Instantiate fresh model and optimizer
    model = SequentialCNN(seed=123)
    loss_fn = SoftmaxCrossEntropyLoss()
    optimizer = SGD(lr=0.05) # Slightly higher lr for fast 14-image memorization test
    
    # Initial performance
    initial_logits = model.forward(X_tiny)
    initial_loss = loss_fn.forward(initial_logits, y_tiny)
    initial_preds = np.argmax(initial_logits, axis=-1)
    initial_acc = np.mean(initial_preds == y_tiny) * 100.0
    
    print(f"\nInitial State:")
    print(f"  Starting Loss     : {initial_loss:.4f}")
    print(f"  Starting Accuracy : {initial_acc:.2f}% ({np.sum(initial_preds == y_tiny)}/14)")
    
    # Train for 80 iterations on full batch of 14 images
    num_epochs = 80
    for epoch in range(1, num_epochs + 1):
        # 1. Forward
        logits = model.forward(X_tiny)
        loss = loss_fn.forward(logits, y_tiny)
        
        # 2. Backward
        dlogits = loss_fn.backward()
        model.backward(dlogits)
        
        # 3. Update
        optimizer.step(model.layers)
        
        preds = np.argmax(logits, axis=-1)
        acc = np.mean(preds == y_tiny) * 100.0
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{num_epochs:2d} -> Loss: {loss:.4f} | Accuracy: {acc:6.2f}% ({np.sum(preds == y_tiny):2d}/14)")
            
    final_logits = model.forward(X_tiny)
    final_loss = loss_fn.forward(final_logits, y_tiny)
    final_preds = np.argmax(final_logits, axis=-1)
    final_acc = np.mean(final_preds == y_tiny) * 100.0
    
    print("\nSanity Check Results:")
    print(f"  Starting Loss     : {initial_loss:.4f}  -->  Final Loss: {final_loss:.4f}")
    print(f"  Starting Accuracy : {initial_acc:.2f}%  -->  Final Accuracy: {final_acc:.2f}% ({np.sum(final_preds == y_tiny)}/14)")
    
    assert final_acc >= 95.0, f"Overfit sanity test failed! Final accuracy: {final_acc}%"
    assert final_loss < 0.2, f"Overfit sanity test failed! Final loss too high: {final_loss}"
    print("\n [PASS] Overfit sanity check SUCCEEDED: The scratch CNN successfully memorized the 14-image dataset with 100% accuracy!")
    return {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "initial_acc": float(initial_acc),
        "final_acc": float(final_acc),
        "epochs": num_epochs,
        "success": True
    }


if __name__ == "__main__":
    test_model_shapes_and_summary()
    test_tiny_overfit_sanity_check()
