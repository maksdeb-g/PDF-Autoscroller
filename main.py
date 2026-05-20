import cv2
import numpy as np
import pyautogui
import time
from gaze_detector import FaceMeshDetector
from gaze_ratio    import get_gaze_ratio
from calibration   import Calibrator, CALIB_POINTS, SCREEN_W, SCREEN_H
from smoother      import GazeSmoother

pyautogui.FAILSAFE = False

def run_calibration(cap, detector, calibrator):
    overlay = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # Warm up camera auto-exposure
    for _ in range(10):
        cap.read()

    for (nx, ny) in CALIB_POINTS:
        px, py = int(nx * SCREEN_W), int(ny * SCREEN_H)
        samples = []

        no_face_frames = 0
        for _ in range(60):   # collect 60 frames (~2 sec at 30fps)
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            lm = detector.get_landmarks(frame)
            if lm is not None:
                h, v = get_gaze_ratio(lm, 'left')
                # Average with right eye for robustness
                hr, vr = get_gaze_ratio(lm, 'right')
                samples.append(((h+hr)/2, (v+vr)/2))
            else:
                no_face_frames += 1

            overlay[:] = 0
            cv2.circle(overlay, (px, py), 20, (0, 255, 0), -1)
            status = "Look at the dot"
            if no_face_frames > 10:
                status += " - FACE NOT VISIBLE"
            cv2.putText(overlay, status, (SCREEN_W//2-200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imshow("Calibration", overlay)
            cv2.waitKey(1)

        if samples:
            avg = np.mean(samples, axis=0)
            calibrator.collect_point((px, py), avg)

    print(f"Calibration points collected: {len(calibrator.screen_pts)}")
    print(f"Gaze samples collected: {len(calibrator.gaze_samples)}")
    calibrator.fit()
    cv2.destroyWindow("Calibration")
    print("Calibration complete.")

def release_camera(cap):
    try:
        cap.release()
    except:
        pass
    cv2.destroyAllWindows()

def main():
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Cannot open camera.")
            print("Try: Close other apps using the camera (Zoom, Discord, browser),")
            print("     then restart this program.")
            return

        detector   = FaceMeshDetector()
        calibrator = Calibrator()
        smoother   = GazeSmoother()

        run_calibration(cap, detector, calibrator)

        # ---- Scroll parameters ----
        DEAD_ZONE   = 0.15   # 15% of screen above/below center = no scroll
        MAX_SPEED   = 60     # max scroll clicks per second
        scroll_acc  = 0.0
        paused      = False
        prev_time   = time.time()
        show_status = 0      # status message timer

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            now = time.time()
            dt  = now - prev_time
            prev_time = now

            # Overlay info
            overlay_text = []
            scroll_speed = 0

            lm = detector.get_landmarks(frame)

            if lm is not None:
                hl, vl = get_gaze_ratio(lm, 'left')
                hr, vr = get_gaze_ratio(lm, 'right')
                h_avg, v_avg = (hl+hr)/2, (vl+vr)/2

                screen_x, screen_y = calibrator.predict(h_avg, v_avg)
                sx, sy = smoother.smooth(screen_x, screen_y)

                # ---- Proportional scroll ----
                if not paused:
                    mid = SCREEN_H / 2
                    dz  = SCREEN_H * DEAD_ZONE
                    if sy < mid - dz:
                        factor = (mid - dz - sy) / (mid - dz)        # 0 → 1
                        factor = factor ** 2
                        scroll_speed = -int(MAX_SPEED * factor)
                    elif sy > mid + dz:
                        factor = (sy - mid - dz) / (SCREEN_H - mid - dz)  # 0 → 1
                        factor = factor ** 2
                        scroll_speed = int(MAX_SPEED * factor)

                    # Accumulate sub-pixel scroll
                    if scroll_speed != 0:
                        scroll_acc += scroll_speed * dt
                        whole = int(scroll_acc)
                        if whole != 0:
                            pyautogui.scroll(whole)
                            scroll_acc -= whole

                overlay_text.append(f"Gaze: ({int(sx)}, {int(sy)})")
                overlay_text.append(f"Scroll: {'PAUSED' if paused else f'{scroll_speed:+d} cps'}")

                # Gaze cursor
                cx = int(np.clip(sx / SCREEN_W * frame.shape[1], 0, frame.shape[1] - 1))
                cy = int(np.clip(sy / SCREEN_H * frame.shape[0], 0, frame.shape[0] - 1))
                cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)
                cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (0, 255, 255), 1)
                cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (0, 255, 255), 1)
            else:
                overlay_text.append("No face detected")

            # ---- Scroll zone indicator (right edge) ----
            h, w = frame.shape[:2]
            bar_x, bar_w = w - 20, 12
            bar_h = h - 40
            bar_y0 = 20
            cv2.rectangle(frame, (bar_x, bar_y0), (bar_x + bar_w, bar_y0 + bar_h), (100, 100, 100), 1)

            dz_px = int(bar_h * DEAD_ZONE)
            mid_px = bar_y0 + bar_h // 2
            cv2.rectangle(frame, (bar_x, mid_px - dz_px), (bar_x + bar_w, mid_px + dz_px), (80, 80, 80), -1)
            # Fill active scroll zone
            if scroll_speed > 0:
                fill = int(bar_h * min(abs(scroll_speed) / MAX_SPEED, 1))
                cv2.rectangle(frame, (bar_x, bar_y0 + bar_h - fill), (bar_x + bar_w, bar_y0 + bar_h), (0, 200, 0), -1)
            elif scroll_speed < 0:
                fill = int(bar_h * min(abs(scroll_speed) / MAX_SPEED, 1))
                cv2.rectangle(frame, (bar_x, bar_y0), (bar_x + bar_w, bar_y0 + fill), (0, 200, 0), -1)

            # ---- Text overlay ----
            for i, txt in enumerate(overlay_text):
                cv2.putText(frame, txt, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ---- Controls hint ----
            cv2.putText(frame, "[Q]uit  [P]ause  [+/-] Speed", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            if show_status > 0:
                show_status -= 1

            cv2.imshow("Gaze Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                scroll_acc = 0
                show_status = 30
            elif key == ord('+') or key == ord('='):
                MAX_SPEED = min(MAX_SPEED + 10, 200)
                show_status = 30
            elif key == ord('-') or key == ord('_'):
                MAX_SPEED = max(MAX_SPEED - 10, 10)
                show_status = 30
    finally:
        release_camera(cap)

if __name__ == "__main__":
    main()
