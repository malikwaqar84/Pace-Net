"""
Explainability Methods for PACE-Net
=====================================
Implements:
  1. SHAP  — global & local feature importance (Lundberg & Lee, 2017)
  2. Grad-CAM — brain ROI saliency maps (Selvaraju et al., 2017)
  3. Integrated Gradients — axiomatic attribution (Sundararajan et al., 2017)

References:
  [29] Lundberg, S. M. & Lee, S.-I. NeurIPS 2017
  [30] Sundararajan, M. et al. ICML 2017
  [31] Selvaraju, R. R. et al. ICCV 2017
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from captum.attr import IntegratedGradients, GradientShap
from captum.attr import LayerGradCam


class SHAPAnalyser:
    """
    SHAP-based global and local feature attribution.
    Uses KernelSHAP on the linearised post-CGAT fusion output.
    """

    def __init__(self, model: nn.Module, background_data: torch.Tensor,
                 feature_names: List[str]):
        """
        Args:
            model:           PACE-Net model
            background_data: background dataset for KernelSHAP
            feature_names:   list of feature names for plotting
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names

    def compute_shap_values(self, input_data: torch.Tensor,
                            n_samples: int = 100) -> np.ndarray:
        """
        Compute SHAP values for input_data.

        Args:
            input_data: (N, n_features) input features
            n_samples:  number of samples for KernelSHAP approximation
        Returns:
            shap_values: (N, n_features) SHAP values
        """
        try:
            import shap
            # Wrap model for SHAP
            def model_predict(x):
                with torch.no_grad():
                    x_t = torch.FloatTensor(x)
                    out = self.model.linear_predict(x_t)
                    return torch.softmax(out, dim=-1).numpy()

            explainer = shap.KernelExplainer(
                model_predict,
                self.background_data.numpy()[:n_samples]
            )
            shap_values = explainer.shap_values(
                input_data.numpy(), nsamples=n_samples
            )
            return np.array(shap_values)
        except ImportError:
            print("SHAP not installed. Run: pip install shap")
            return np.zeros((input_data.shape[0], len(self.feature_names)))

    def global_importance(self, shap_values: np.ndarray) -> Dict[str, float]:
        """
        Compute global feature importance as mean |SHAP|.

        Args:
            shap_values: (n_classes, N, n_features) SHAP values
        Returns:
            importance: {feature_name: mean_abs_shap}
        """
        # Average over classes and samples
        if shap_values.ndim == 3:
            mean_abs = np.abs(shap_values).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        importance = {
            self.feature_names[i]: float(mean_abs[i])
            for i in range(min(len(self.feature_names), len(mean_abs)))
        }
        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def patient_waterfall(self, shap_values: np.ndarray,
                          base_value: float,
                          subject_idx: int = 0) -> Dict:
        """
        Generate patient-level waterfall plot data.
        Returns cumulative SHAP contributions for a single subject.
        """
        if shap_values.ndim == 3:
            sv = shap_values[2, subject_idx]  # AD class SHAP
        else:
            sv = shap_values[subject_idx]

        sorted_idx = np.argsort(np.abs(sv))[::-1]
        waterfall = {
            'base_value': base_value,
            'features': [self.feature_names[i] for i in sorted_idx],
            'shap_values': sv[sorted_idx].tolist(),
            'cumulative': np.cumsum(
                np.concatenate([[base_value], sv[sorted_idx]])
            ).tolist()
        }
        return waterfall


class GradCAMAnalyser:
    """
    Grad-CAM attribution for brain ROI saliency.
    Applied to BCG Transformer node features and structural equations.
    """

    def __init__(self, model: nn.Module, target_layer: str = 'bcg_transformer'):
        self.model = model
        self.target_layer = target_layer
        self.gradients = {}
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        def get_gradient(name):
            def hook(module, grad_in, grad_out):
                self.gradients[name] = grad_out[0].detach()
            return hook

        # Find target layer in model
        for name, module in self.model.named_modules():
            if self.target_layer in name and hasattr(module, 'weight'):
                module.register_forward_hook(get_activation(name))
                module.register_backward_hook(get_gradient(name))

    def compute_roi_saliency(self, batch: dict,
                              target_class: int = 2,
                              n_rois: int = 200) -> np.ndarray:
        """
        Compute per-ROI Grad-CAM saliency for a target class.

        Args:
            batch:        model input batch
            target_class: 0=CN, 1=MCI, 2=AD
            n_rois:       number of brain ROIs
        Returns:
            saliency: (n_rois,) normalised saliency scores
        """
        self.model.eval()
        self.model.zero_grad()

        logits, _, _, _ = self.model(batch, return_attention=False)
        logits[:, target_class].sum().backward()

        # Get activation and gradient for target layer
        layer_keys = list(self.activations.keys())
        if not layer_keys:
            return np.zeros(n_rois)

        key = [k for k in layer_keys if self.target_layer in k]
        if not key:
            return np.zeros(n_rois)
        key = key[0]

        act = self.activations[key]    # (N_rois, dim)
        grad = self.gradients.get(key, torch.ones_like(act))

        # Grad-CAM: weight activations by mean gradient
        weights = grad.mean(0)         # (dim,)
        cam = (act * weights).sum(-1)  # (N_rois,)
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def equation_attribution(self, batch: dict) -> Dict[str, float]:
        """
        Compute gradient-based attribution per structural equation.
        Returns attribution score for each of Equations (6)-(10).
        """
        self.model.eval()
        self.model.zero_grad()

        logits, _, _, aux = self.model(batch, return_attention=True)
        logits.sum().backward()

        attributions = {}
        for eq_name in ['eq_G', 'eq_A', 'eq_C']:
            for name, param in self.model.named_parameters():
                if eq_name in name and param.grad is not None:
                    attr = param.grad.abs().mean().item()
                    attributions[eq_name] = attributions.get(eq_name, 0) + attr
        return attributions


class IntegratedGradientsAnalyser:
    """
    Integrated Gradients with axiomatic attribution guarantees.
    (Completeness + Implementation Invariance)
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        """
        Args:
            model:   PACE-Net model (wrapped for IG)
            n_steps: number of interpolation steps (50 in paper)
        """
        self.model = model
        self.n_steps = n_steps

    def compute_attribution(self, x: torch.Tensor,
                             baseline: Optional[torch.Tensor] = None,
                             target_class: int = 2) -> np.ndarray:
        """
        Compute Integrated Gradients attribution.

        IG_i(x) = (x_i - x'_i) * ∫_0^1 ∂F(x' + α(x-x'))/∂x_i dα

        Args:
            x:            (B, n_features) input features
            baseline:     (B, n_features) baseline (zeros if None)
            target_class: class to attribute to
        Returns:
            attributions: (B, n_features) IG attributions
        """
        if baseline is None:
            baseline = torch.zeros_like(x)

        attributions = torch.zeros_like(x)
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            x_interp = baseline + alpha * (x - baseline)
            x_interp.requires_grad_(True)

            # Forward pass
            out = self.model.forward_features(x_interp)
            logits = self.model.classifier_head(out)
            target_score = logits[:, target_class].sum()

            # Backward
            target_score.backward(retain_graph=True)
            if x_interp.grad is not None:
                attributions += x_interp.grad.detach()
            self.model.zero_grad()

        # Riemann approximation: multiply by (x - baseline)
        ig = (x - baseline) * attributions / self.n_steps
        return ig.numpy()

    def attribution_trajectory(self, subjects_by_class: Dict[int, torch.Tensor],
                                feature_names: List[str]) -> Dict:
        """
        Compute attribution trajectory across disease spectrum.
        Returns mean attribution per feature per class (CN→MCI→AD).
        """
        trajectory = {}
        class_names = {0: 'CN', 1: 'MCI', 2: 'AD'}
        for cls, x_cls in subjects_by_class.items():
            attrs = self.compute_attribution(x_cls, target_class=cls)
            mean_abs = np.abs(attrs).mean(0)
            trajectory[class_names[cls]] = {
                feat: float(val) for feat, val in zip(feature_names, mean_abs)
            }
        return trajectory
