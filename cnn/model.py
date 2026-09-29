"""
Sequential CNN Model Class
==========================
Pure NumPy model container managing forward propagation, backpropagation,
parameter tracking, model summary, and checkpointing.
"""

from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np

from .base import Layer
from .layers import Conv2D, MaxPool2D, Flatten, Dense
from .activations import ReLU, Softmax


class SequentialCNN:
    """
    Sequential Container for Convolutional Neural Network architectures.
    
    Default Architecture for Indian Currency Classification:
        Input: (N, 3, 64, 64)
        1. Conv2D(3 -> 8, kernel=3, stride=1, padding=1)  -> (N, 8, 64, 64)
        2. ReLU()                                           -> (N, 8, 64, 64)
        3. MaxPool2D(pool=2, stride=2)                     -> (N, 8, 32, 32)
        4. Conv2D(8 -> 16, kernel=3, stride=1, padding=1) -> (N, 16, 32, 32)
        5. ReLU()                                           -> (N, 16, 32, 32)
        6. MaxPool2D(pool=2, stride=2)                     -> (N, 16, 16, 16)
        7. Flatten()                                        -> (N, 4096)
        8. Dense(4096 -> 32)                                -> (N, 32)
        9. ReLU()                                           -> (N, 32)
        10. Dense(32 -> 7)                                  -> (N, 7)
    """
    def __init__(self, layers: Optional[List[Layer]] = None, seed: int = 42):
        self.seed = seed
        self.softmax = Softmax()
        
        if layers is not None:
            self.layers = layers
        else:
            self.layers = self._build_default_architecture(seed)

    def _build_default_architecture(self, seed: int) -> List[Layer]:
        """Constructs the baseline CNN architecture."""
        rng = np.random.default_rng(seed)
        s1, s2, s3, s4 = rng.integers(0, 100000, size=4)
        
        return [
            # Block 1
            Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1, seed=int(s1)),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            
            # Block 2
            Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1, seed=int(s2)),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            
            # Classifier Head
            Flatten(),
            Dense(in_features=16 * 16 * 16, out_features=32, seed=int(s3)),
            ReLU(),
            Dense(in_features=32, out_features=7, seed=int(s4))
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Propagates input tensor through all layers sequentially.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            Logits tensor of shape (N, num_classes).
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Propagates loss gradient backwards through all layers in reverse order.
        
        Args:
            grad_output: Upstream gradient dL/dLogits of shape (N, num_classes).
            
        Returns:
            Input gradient dL/dX.
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Returns predicted class indices for input batch.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            1D array of class integer labels of shape (N,).
        """
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Returns predicted normalized class probabilities for input batch.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            2D array of class probabilities of shape (N, num_classes).
        """
        logits = self.forward(x)
        return self.softmax.forward(logits)

    def parameters(self) -> List[Tuple[str, np.ndarray]]:
        """Returns list of (param_name, param_array) tuples for all trainable parameters."""
        params_list = []
        for l_idx, layer in enumerate(self.layers):
            if getattr(layer, "trainable", False) and hasattr(layer, "params"):
                for name, param in layer.params.items():
                    params_list.append((f"layer_{l_idx}_{layer.__class__.__name__}_{name}", param))
        return params_list

    def gradients(self) -> List[Tuple[str, np.ndarray]]:
        """Returns list of (param_name, grad_array) tuples for all trainable gradients."""
        grads_list = []
        for l_idx, layer in enumerate(self.layers):
            if getattr(layer, "trainable", False) and hasattr(layer, "grads"):
                for name, grad in layer.grads.items():
                    grads_list.append((f"layer_{l_idx}_{layer.__class__.__name__}_{name}", grad))
        return grads_list

    def count_parameters(self) -> Dict[str, int]:
        """Calculates total, trainable, and per-layer parameter counts."""
        total_params = 0
        layer_breakdown = []
        
        for l_idx, layer in enumerate(self.layers):
            layer_params = 0
            if getattr(layer, "trainable", False) and hasattr(layer, "params"):
                for name, param in layer.params.items():
                    layer_params += param.size
            total_params += layer_params
            layer_breakdown.append({
                "layer_idx": l_idx,
                "layer_type": layer.__class__.__name__,
                "params": layer_params
            })
            
        return {
            "total_parameters": total_params,
            "trainable_parameters": total_params,
            "layer_breakdown": layer_breakdown
        }

    def summary(self, input_shape: Tuple[int, ...] = (1, 3, 64, 64)) -> str:
        """
        Generates a formatted architectural summary table with output shapes and parameter counts.
        """
        lines = []
        lines.append("=" * 75)
        lines.append(f"{'Layer (type)':<25} | {'Output Shape':<22} | {'Param #':<10}")
        lines.append("=" * 75)
        
        # Test pass with dummy input to trace shapes
        dummy_x = np.zeros(input_shape, dtype=np.float64)
        current_x = dummy_x
        total_params = 0
        
        for l_idx, layer in enumerate(self.layers):
            current_x = layer.forward(current_x)
            layer_name = f"{l_idx+1}. {layer.__class__.__name__}"
            out_shape_str = str(current_x.shape)
            
            p_count = 0
            if getattr(layer, "trainable", False) and hasattr(layer, "params"):
                for p in layer.params.values():
                    p_count += p.size
            total_params += p_count
            
            lines.append(f"{layer_name:<25} | {out_shape_str:<22} | {p_count:<10,}")
            lines.append("-" * 75)
            
        lines.append(f"Total Trainable Parameters: {total_params:,}")
        lines.append("=" * 75)
        return "\n".join(lines)

    def save_weights(self, filepath: Union[str, Path]):
        """Saves all learnable weights and biases to a compressed NumPy (.npz) file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        weights_dict = {}
        for l_idx, layer in enumerate(self.layers):
            if getattr(layer, "trainable", False) and hasattr(layer, "params"):
                for name, param in layer.params.items():
                    weights_dict[f"layer_{l_idx}_{name}"] = param
                    
        np.savez_compressed(str(path), **weights_dict)

    def load_weights(self, filepath: Union[str, Path]):
        """Loads learnable weights and biases from a compressed NumPy (.npz) file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Weight file not found: {path}")
            
        data = np.load(str(path))
        for l_idx, layer in enumerate(self.layers):
            if getattr(layer, "trainable", False) and hasattr(layer, "params"):
                for name in layer.params:
                    key = f"layer_{l_idx}_{name}"
                    if key in data:
                        layer.params[name][:] = data[key]
                        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
