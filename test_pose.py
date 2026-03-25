#!/usr/bin/env python3
"""
Test script for pose estimation and gait analysis.
"""

from models.inference import PoseEstimation
from utils.metrics import GaitData
from models.architecture import AnomalyDetector
import os

def test_pose():
    # Test on a sample image if available
    pose = PoseEstimation()
    test_img = "data/raw/test/test_image.jpg"
    if pose.model is None:
        print("[SKIP] Pose model not loaded. Please download yolov8n-pose.pt and place it in the project root or set the correct path in PoseEstimation.")
        return
    if not os.path.exists(test_img):
        print(f"[SKIP] Test image not found at {test_img}. Please add a test image.")
        return
    try:
        keypoints = pose.estimate_pose(test_img)
        if keypoints is not None:
            print("Keypoints shape:", keypoints.shape)
        else:
            print("Pose estimation ran, but no keypoints detected.")
    except Exception as e:
        print(f"[ERROR DETECTED] Exception during pose estimation: {e}")

def test_gait():
    # Test gait data loading (need a keypoints csv)
    csv_path = "sample_keypoints.csv"
    if os.path.exists(csv_path):
        gait = GaitData(csv_path)
        features = gait.extract_features()
        print("Features:", features)
    else:
        print("No keypoints CSV found")

def test_anomaly():
    # Test anomaly detector
    detector = AnomalyDetector()
    # Sample features
    features = [
        {'stride_length': 10, 'head_bobbing': 5, 'symmetry': 1, 'cadence': 2},
        {'stride_length': 12, 'head_bobbing': 3, 'symmetry': 0.5, 'cadence': 2.1}
    ]
    detector.fit(features)
    score = detector.predict(features[0])
    print("Anomaly score:", score)

if __name__ == "__main__":
    test_pose()
    # test_gait()
    # test_anomaly()