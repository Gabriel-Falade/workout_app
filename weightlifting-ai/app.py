import os
import sys
import time
import tempfile
from typing import List, Optional, Dict
import cv2 as cv
import mediapipe as mp
import numpy as np
import streamlit as st

# --- Make 'src' importable when running from project root ---
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Project imports
from analysis.pose_helpers import mp_results_to_dict
from analysis.frame_metrics import compute_frame_metrics, FrameMetricsState
from exercises.pushup import PushUpDetector

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

def class_to_color(rep_class: str):
    """Map rep classification to a BGR color."""
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
# Video Processing Function
# ------------------------------
def process_video(video_path, st_video_placeholder, st_info_placeholder, st_progress_bar):
    """Process video file frame by frame."""
    video_capture = cv.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        st.error("Failed to open video file")
        return
    
    # Get video properties
    total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv.CAP_PROP_FPS)
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    push = PushUpDetector()
    fm_state = FrameMetricsState()
    
    fps_ema = None
    prev = time.time()
    frame_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while video_capture.isOpened():
            ok, frame = video_capture.read()
            if not ok:
                st.success("Video processing completed!")
                break

            frame_count += 1
            
            # Update progress bar
            progress = frame_count / total_frames
            st_progress_bar.progress(progress)

            # Process frame
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

            # Convert to RGB for Streamlit and display
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            st_video_placeholder.image(image, channels="RGB", use_column_width=True)

            # Display real-time info below the video
            rom_val = rom if rom is not None else 0.0
            vel_val = vel if vel is not None else 0.0
            info_text = f"""
            **Current Stage**: {live["stage"] or '--'}  
            **Total Reps**: {live["rep_count"]}  
            **ROM**: {rom_val:.1f}%  
            **Velocity**: {vel_val:.1f}%/s
            """
            st_info_placeholder.markdown(info_text)
            
            # Control playback speed (optional)
            time.sleep(1.0 / fps if fps > 0 else 0.033)

    video_capture.release()

# ------------------------------
# Main Streamlit App
# ------------------------------
def main():
    st.set_page_config(layout="wide", page_title="AI Workout Analyzer")
    st.title("🏋️ AI-Powered Workout App")
    
    st.sidebar.header("⚙️ Settings")
    
    source = st.sidebar.radio("Video Source", ("Upload Video File", "Webcam (Not Yet Supported)"))
    
    if source == "Webcam (Not Yet Supported)":
        st.warning("⚠️ Webcam support requires streamlit-webrtc or a custom implementation.")
        st.info("For now, please use the 'Upload Video File' option or run the standalone test.py script for webcam support.")
        return
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is None:
        st.info("👆 Please upload a video file to begin analysis")
        return
    
    st.sidebar.success("✅ File uploaded successfully!")
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    st.header("📊 Workout Analysis")
    
    # Create placeholders
    st_video_placeholder = st.empty()
    st_progress_bar = st.progress(0)
    st_info_placeholder = st.empty()
    
    # Add a start button
    if st.sidebar.button("▶️ Start Analysis", type="primary"):
        try:
            process_video(video_path, st_video_placeholder, st_info_placeholder, st_progress_bar)
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)

if __name__ == "__main__":
    main()
