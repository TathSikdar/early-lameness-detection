import cv2
import os
import numpy as np
from ultralytics import YOLO
from models.pose_analysis import compute_lameness_score

def extract_frames_from_video(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and count >= max_frames):
            break
        frames.append(frame)
        count += 1
    cap.release()
    return frames

def test_pose_analysis_on_top_video():
    video_path = "data/raw/top/top_better_quality.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return
    print(f"Extracting frames from {video_path}...")
    frames = extract_frames_from_video(video_path, max_frames=100)
    print(f"Extracted {len(frames)} frames.")
    print("Loading YOLOv26 pose model...")
    model = YOLO("yolov8n-pose.pt")
    print("Running lameness scoring...")
    result = compute_lameness_score(frames, model)
    print("\nLameness scoring result:")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    test_pose_analysis_on_top_video()
