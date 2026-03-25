from utils.video import Segmenter
from extract_frames import Extracter
from extract_frames import Cleaner
from models.inference import PoseEstimation
from utils.metrics import GaitData
from models.architecture import AnomalyDetector
import json
import os
import subprocess 

# extract = Extracter(
#     output_dir="data/processed",
#     session_id="session_1"
# )
# seg = Segmenter()


# metadata_path = seg.run_full_pipeline(
#     session_id="session_1",
#     output_dir="data/processed",
#     background_frame_numbers={'top': 0, 'front': 0},  # Side doesn't need background
#     visualize=False,
#     views_to_visualize=['top']  # Only visualize side view as in original
# )
# print("++++++++++++++++++++++++++++++++++++++++++++++++")
# print("All meetadata extracted from file, loading in extracter")

# print("++++++++++++++++++++++++++++++++++++++++++++++++")
# print("Extracter loaded in, running Extracter")
# extract.extract_all_segments()
# print("++++++++++++++++++++++++++++++++++++++++++++++++")
# print("Extracted done running")

# print("++++++++++++++++++++++++++++++++++++++++++++++++")
# print("Running pose estimation on all cows")

# Load metadata
metadata_path = "data/processed/session_1/metadata_session_1.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

pose_estimator = PoseEstimation()
if pose_estimator.model is None:
    raise RuntimeError("Pose estimation model is not loaded. Please download yolov8n-pose.pt or set a valid model_path in PoseEstimation.")

anomaly_detector = AnomalyDetector()
all_features = []

all_keys = list(metadata.keys())
all_cows = list(metadata[all_keys[2]].keys()) #Gets the cow ids from the first session (assuming all sessions have same cows)
for cow_id in all_cows:
    print(f"Processing cow {cow_id}")
    # Use top view video
    top_video_path = f"data/processed/session_1/{cow_id}/top/{cow_id}_top_segment.mp4"
    if not os.path.exists(top_video_path):
        print(f"Video not found for {cow_id}")
        continue
    
    # Estimate pose and save keypoints
    keypoints_csv = f"data/processed/session_1/{cow_id}/keypoints.csv"
    pose_estimator.estimate_pose_video(top_video_path, keypoints_csv)
    
    # Extract gait features
    gait = GaitData(keypoints_csv)
    features = gait.extract_features()
    all_features.append(features)

# Train anomaly detector on all cows (assuming most are healthy)
anomaly_detector.fit(all_features)

# Save model
anomaly_detector.save_model("models/anomaly_model.pkl")

print("Anomaly model trained and saved")

print("++++++++++++++++++++++++++++++++++++++++++++++++")
print("Loading dashboard to prompt for corrections!")
subprocess.run("streamlit run review_dashboard.py", shell=True) #Runs command in terminal

# Uncomment to clean all cows
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# clen = Cleaner(
#     output_dir="data/processed",
#     session_id="session_1"
# )
# clen.remove_all_cows()
