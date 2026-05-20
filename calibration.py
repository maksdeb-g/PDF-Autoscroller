import numpy as np
import cv2
import time

SCREEN_W, SCREEN_H = 1920, 1080

# 9 calibration points (normalized 0-1)
CALIB_POINTS = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]

class Calibrator:
    def __init__(self):
        self.gaze_samples = []   # list of (h_ratio, v_ratio)
        self.screen_pts   = []   # list of (screen_x, screen_y)
        self.model_x = None
        self.model_y = None

    def collect_point(self, screen_pt, gaze_ratios):
        """Call this after averaging ~30 frames of gaze at screen_pt."""
        self.screen_pts.append(screen_pt)
        self.gaze_samples.append(gaze_ratios)

    def fit(self):
        G = np.array(self.gaze_samples, dtype=np.float64)
        S = np.array(self.screen_pts, dtype=np.float64).reshape(-1, 2)  # force 2D

        if len(G) < 6:
            raise RuntimeError(
                f"Not enough calibration samples ({len(G)}). "
                "Make sure your face is visible during calibration.")

        def features(h, v):
            return [1, h, v, h*h, h*v, v*v]

        F = np.array([features(h, v) for h, v in G])

        self.model_x, _, _, _ = np.linalg.lstsq(F, S[:, 0], rcond=None)
        self.model_y, _, _, _ = np.linalg.lstsq(F, S[:, 1], rcond=None)

    def predict(self, h_ratio, v_ratio):
        f = np.array([1, h_ratio, v_ratio,
                      h_ratio**2, h_ratio*v_ratio, v_ratio**2])
        x = float(f @ self.model_x)
        y = float(f @ self.model_y)
        return (np.clip(x, 0, SCREEN_W), np.clip(y, 0, SCREEN_H))
    