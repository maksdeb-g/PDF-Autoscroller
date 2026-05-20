import cv2
import numpy as np

class GazeSmoother:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # state: [x, y, vx, vy], obs: [x, y]
        self.kf.measurementMatrix = np.array(
            [[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov     *= 0.03
        self.kf.measurementNoiseCov *= 1
        self._initialized = False

    def smooth(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if not self._initialized:
            self.kf.statePre = np.array(
                [[x],[y],[0],[0]], np.float32)
            self._initialized = True
        self.kf.correct(measurement)
        predicted = self.kf.predict()
        return float(predicted[0, 0]), float(predicted[1, 0])