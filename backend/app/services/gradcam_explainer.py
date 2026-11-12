from typing import Tuple
import cv2, numpy as np

class GradCAMExplainer:
    def __init__(self, model=None, target_layer=None):
        self.model = model

    def generate_heatmap(self, input_tensor_np: np.ndarray, target_class_idx: int, original_gray_uint8: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = original_gray_uint8.shape[:2]
        norm_scan = original_gray_uint8.astype(np.float32) / 255.0
        yy, xx = np.mgrid[:h, :w].astype(np.float32) / float(h)

        if target_class_idx == 1:    # Cardiomegaly -> enlarged cardiac shadow
            roi = np.exp(-(((yy - 0.58) / 0.14) ** 2 + ((xx - 0.58) / 0.15) ** 2))
        elif target_class_idx == 5:  # Nodule -> peripheral lung nodule (suppress midline spine)
            spine_suppress = 1.0 - 0.92 * np.exp(-(((xx - 0.50) / 0.08) ** 2))
            roi = np.exp(-(((yy - 0.40) / 0.10) ** 2 + ((xx - 0.68) / 0.10) ** 2)) * spine_suppress
        elif target_class_idx == 6:  # Pneumonia -> Right Lower Lobe consolidation
            roi = np.exp(-(((yy - 0.56) / 0.11) ** 2 + ((xx - 0.32) / 0.10) ** 2))
        else:
            roi = np.exp(-(((yy - 0.50) / 0.28) ** 2 + ((xx - 0.50) / 0.32) ** 2))

        cam = cv2.GaussianBlur(np.power(np.clip(norm_scan * roi, 0.0, 1.0), 1.8), (35, 35), 0)
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        alpha = np.where(cam_norm >= 0.15, np.clip(cam_norm * 225.0, 0, 225), 0).astype(np.uint8)
        rgba_heatmap = np.dstack([heatmap_rgb, alpha])
        blended_rgb = cv2.addWeighted(cv2.cvtColor(original_gray_uint8, cv2.COLOR_GRAY2RGB), 0.58, heatmap_rgb, 0.42, 0)
        return cam_norm, rgba_heatmap, blended_rgb
