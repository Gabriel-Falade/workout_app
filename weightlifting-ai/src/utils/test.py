# test.py — live push-up detector with feedback overlay
import os, sys, time
from typing import List, Optional, Dict
import cv2 as cv
import mediapipe as mp
import numpy as np

# --- Make 'src' importable when running from src/utils ---
THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))  # .. = src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Project imports
from analysis.pose_helpers import mp_results_to_dict
from analysis.frame_metrics import compute_frame_metrics, FrameMetricsState
from exercises.pushup import PushUpDetector

# ------------------------------
# Camera helper: try multiple indices/backends on Windows
# ------------------------------
def open_camera():
    indices = [0, 1, 2]
    backends = [cv.CAP_DSHOW, cv.CAP_MSMF, cv.CAP_ANY]
    for i in indices:
        for be in backends:
            cap = cv.VideoCapture(i, be)
            if cap.isOpened():
                return cap
            cap.release()
    return None

# ------------------------------
# UI helpers
# ------------------------------
def draw_info_box(img, lines: List[str], padding=10, line_height=22):
    """Bottom-right semi-opaque box with lines of text."""
    if img is None or not lines:
        return img
    h, w = img.shape[:2]
    font = cv.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    thickness = 1

    sizes = [cv.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
    text_w = max((sz[0] for sz in sizes), default=0)
    box_w = text_w + padding * 2
    box_h = line_height * len(lines) + padding * 2

    x1 = max(0, w - box_w - 10)
    y1 = max(0, h - box_h - 10)
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = img.copy()
    cv.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    img = cv.addWeighted(overlay, 0.65, img, 0.35, 0)

    y = y1 + padding + 15
    for s in lines:
        cv.putText(img, s, (x1 + padding, y), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
        y += line_height
    return img

def draw_toast(img, lines: List[str], color=(60, 180, 75), duration_ms=1600, state: Optional[Dict]=None):
    """
    Bottom-center toast shown for duration_ms after it's (re)armed in state['until_ts'].
    - state: dict holding {'until_ts': float} in seconds.
    - color: BGR tuple for banner background.
    """
    if img is None or not lines or state is None:
        return img

    now = time.time()
    until = state.get("until_ts", 0.0)
    if now > until:
        return img  # expired / not armed

    h, w = img.shape[:2]
    font = cv.FONT_HERSHEY_COMPLEX
    fs, th = 0.7, 2

    sizes = [cv.getTextSize(s, font, fs, th)[0] for s in lines]
    text_w = max((sz[0] for sz in sizes), default=0)
    line_h = 28
    pad_x, pad_y = 16, 14

    box_w = text_w + pad_x * 2
    box_h = line_h * len(lines) + pad_y * 2

    x1 = max(10, (w - box_w) // 2)
    y2 = h - 10
    y1 = y2 - box_h

    # BG with slight transparency
    overlay = img.copy()
    cv.rectangle(overlay, (x1, y1), (x1 + box_w, y2), color, -1)
    img = cv.addWeighted(overlay, 0.7, img, 0.3, 0)

    y = y1 + pad_y + 18
    for s in lines:
        cv.putText(img, s, (x1 + pad_x, y), font, fs, (255, 255, 255), th, cv.LINE_AA)
        y += line_h

    return img

def arm_toast(state: Dict, show_ms: int = 1600):
    state["until_ts"] = time.time() + (show_ms / 1000.0)

def class_to_color(rep_class: str):
    """
    Map rep classification to a BGR color.
    good -> green; form_warn -> yellow; depth/lockout fails -> red
    """
    rc = (60, 180, 75)   # green
    if not rep_class:
        return rc
    rc_lower = rep_class.lower()
    if "fail" in rc_lower:
        return (40, 40, 220)     # red
    if "warn" in rc_lower:
        return (0, 215, 255)     # yellow
    return rc

# ------------------------------
# Main
# ------------------------------
def main():
    cap = open_camera()
    if cap is None:
        print("ERROR: Could not open any camera (tried indices 0..2 with DirectShow/MSMF).")
        return

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    push = PushUpDetector()
    fm_state = FrameMetricsState()

    # toast/feedback state
    toast = {"until_ts": 0.0}
    fps_ema = None
    prev = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARN: Failed to read frame.")
                break

            # Pose
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)

            image = cv.cvtColor(image_rgb, cv.COLOR_RGB2BGR)
            image.flags.writeable = True

            # Landmarks -> dict
            lm = mp_results_to_dict(results)

            # Timing
            now = time.time()
            dt = max(1e-3, now - prev)
            prev = now
            fps_inst = 1.0 / dt
            fps_ema = fps_inst if fps_ema is None else (0.2 * fps_inst + 0.8 * fps_ema)

            # Metrics
            fm = compute_frame_metrics(lm, dt, fm_state)

            # Push-up detector
            rep_event, live = push.update(fm, now_s=now)

            # Draw skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # Mirror
            image = cv.flip(image, 1)

            # Top-left: reps & stage
            cv.rectangle(image, (0, 0), (240, 74), (245, 117, 16), -1)
            cv.putText(image, 'REPS', (15, 14), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(image, str(live["rep_count"]), (10, 62), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(image, 'STAGE', (88, 14), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(image, str(live["stage"] or "--"), (80, 62), cv.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

            # Bottom-right: calculations box
            rom = fm.get("rom_pushup_smooth")
            vel = fm.get("vel_pushup")
            tilt = fm.get("torso_tilt_deg")
            info_lines = [
                f"ROM:  {rom:5.1f} %" if rom is not None else "ROM:  --",
                f"Vel:  {vel:5.1f} %/s" if vel is not None else "Vel:  --",
                f"Tilt: {tilt:4.1f} deg" if tilt is not None else "Tilt: --",
                f"FPS:  {fps_ema:4.1f}" if fps_ema is not None else "FPS:  --",
                "Mode: Push-up detector",
            ]
            image = draw_info_box(image, info_lines)

            # Rep feedback toast
            if rep_event:
                # Build lines and arm the toast timer
                rep_idx = rep_event["rep_index_valid"]
                rep_cls = rep_event["class"]
                cues = rep_event.get("cues", [])
                header = f"Rep {rep_idx}: {rep_cls.upper()}"
                body = cues[0] if len(cues) > 0 else ""
                body2 = cues[1] if len(cues) > 1 else ""
                toast["lines"] = [header] + ([body] if body else []) + ([body2] if body2 else [])
                toast["color"] = class_to_color(rep_cls)
                arm_toast(toast, show_ms=1600)

                # Optional: also print the snapshot for debugging
                # print(rep_event)

            # Always draw toast if armed
            image = draw_toast(image, toast.get("lines", []), color=toast.get("color", (60,180,75)), state=toast)

            cv.imshow("Webcam", image)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

