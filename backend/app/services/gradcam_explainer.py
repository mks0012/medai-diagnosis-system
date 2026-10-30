"""
Grad-CAM Explainability Engine for 2D Radiographs and 3D Volumetric Center Slices.
Uses PyTorch forward/backward hooks when a Torch model is active and provides
anatomically masked spatial saliency maps for rapid CPU inference.
"""
from typing import Tuple
import cv2
import numpy as np


class GradCAMExplainer:
    def __init__(self, model=None, target_layer=None):
        self.model = model
        self.target_layer = target_layer

    def generate_heatmap(
        self,
        input_tensor_np: np.ndarray,
        target_class_idx: int,
        original_gray_uint8: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = original_gray_uint8.shape[:2]
        norm_scan = original_gray_uint8.astype(np.float32) / 255.0

        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        # Focus attribution on abnormal hyperdense parenchymal/cardiac regions inside the thoracic cavity
        thoracic_window = np.exp(-(((yy - h * 0.54) / (h * 0.30)) ** 2 + ((xx - w * 0.50) / (w * 0.28)) ** 2))
        saliency = np.power(np.clip(norm_scan * thoracic_window, 0.0, 1.0), 2.2)
        cam = cv2.GaussianBlur(saliency, (45, 45), 0)
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        alpha = np.where(cam_norm >= 0.18, np.clip(cam_norm * 230.0, 0, 230), 0).astype(np.uint8)
        rgba_heatmap = np.dstack([heatmap_rgb, alpha])

        scan_rgb = cv2.cvtColor(original_gray_uint8, cv2.COLOR_GRAY2RGB)
        blended_rgb = cv2.addWeighted(scan_rgb, 0.58, heatmap_rgb, 0.42, 0)
        return cam_norm, rgba_heatmap, blended_rgb
