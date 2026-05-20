import numpy as np

def get_gaze_ratio(landmarks, eye='left'):
    """
    Returns (h_ratio, v_ratio):
      h_ratio: 0 = looking far left, 1 = looking far right
      v_ratio: 0 = looking up,       1 = looking down
    """
    if eye == 'left':
        corner_l = landmarks[33]   # left corner
        corner_r = landmarks[133]  # right corner
        lid_top   = landmarks[159]
        lid_bot   = landmarks[145]
        iris_idx  = [474, 475, 476, 477]
    else:
        corner_l = landmarks[362]
        corner_r = landmarks[263]
        lid_top   = landmarks[386]
        lid_bot   = landmarks[374]
        iris_idx  = [469, 470, 471, 472]

    iris_center = landmarks[iris_idx].mean(axis=0)

    eye_width  = np.linalg.norm(corner_r - corner_l)
    eye_height = np.linalg.norm(lid_bot - lid_top)

    # Project iris onto horizontal axis
    h_ratio = np.dot(iris_center - corner_l, corner_r - corner_l) / (eye_width ** 2)
    v_ratio = np.dot(iris_center - lid_top,  lid_bot  - lid_top)  / (eye_height ** 2)

    return float(np.clip(h_ratio, 0, 1)), float(np.clip(v_ratio, 0, 1))