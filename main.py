from dataclasses import dataclass
import os
import subprocess

from extract_frames import Extracter
from models.inference import PoseEstimation
from models.pose_analysis import compute_lameness_score
from quick_label import Predict
from utils.metrics import GaitData
from utils.video import Segmenter
from detect_cows import create_or_update_cow_json, process_pose_and_lameness_for_cow


@dataclass
class PipelineConfig:
    session_id: str = "session_1"
    output_dir: str = "data/processed"
    top_video_path: str = "data/raw/top/top.mp4"
    side_video_path: str = "data/raw/side/side.mp4"
    front_video_path: str = "data/raw/front/front.mp4"
    tag_model_path: str = "runs/detect/Ear_Tag_Detection_Model/Epochs_50/weights/best.pt"
    row_model_path: str = "runs/obb/Last_Row_Detection_Model/Epochs_50/weights/best.pt"
    pose_model_path: str = "runs/pose/runs/pose/cow_pose_finetune/train6/weights/best.pt"


def run_stage_segmentation(config: PipelineConfig) -> str:
    """Stage 1: Analyze raw footage and write session metadata."""
    segmenter = Segmenter(
        top_path=config.top_video_path,
        side_path=config.side_video_path,
        front_path=config.front_video_path,
    )

    metadata_path = segmenter.run_full_pipeline(
        session_id=config.session_id,
        output_dir=config.output_dir,
        background_frame_numbers={"top": 0, "front": 0},
        visualize=False,
        views_to_visualize=["top"],
    )
    print(f"[Stage 1] Metadata created: {metadata_path}")
    return metadata_path


def run_stage_extract_clips(config: PipelineConfig) -> None:
    """Stage 2: Read metadata and extract per-cow view clips. Also create initial JSON for each cow."""
    extracter = Extracter(output_dir=config.output_dir, session_id=config.session_id)
    stats = extracter.extract_all_segments()
    print(f"[Stage 2] Extraction summary: {stats}")
    # For each cow folder, create initial JSON
    session_dir = os.path.join(config.output_dir, config.session_id)
    if os.path.exists(session_dir):
        for cow_name in os.listdir(session_dir):
            cow_folder = os.path.join(session_dir, cow_name)
            if not os.path.isdir(cow_folder) or not cow_name.startswith("cow_"):
                continue
            # Find top video if exists
            top_dir = os.path.join(cow_folder, "top")
            top_video = None
            if os.path.exists(top_dir):
                for f in os.listdir(top_dir):
                    if f.endswith(".mp4"):
                        top_video = os.path.join(top_dir, f)
                        break
            create_or_update_cow_json(
                cow_id=cow_name,
                session_id=config.session_id,
                video_path=top_video if top_video else "",
                cow_folder=cow_folder
            )
            print(f"[Stage 2] Created/updated lameness_analysis.json for {cow_name}")


def run_stage_ear_tag_metadata(config: PipelineConfig, cow_temp_id: str, predictor: Predict) -> str:
    """Stage 3: Write canonical ear-tag metadata for a cow clip."""
    side_clip = os.path.join(
        config.output_dir,
        config.session_id,
        cow_temp_id,
        "side",
        f"{cow_temp_id}_side_segment.mp4",
    )
    if not os.path.exists(side_clip):
        raise FileNotFoundError(f"Missing side clip for {cow_temp_id}: {side_clip}")

    metadata_path = predictor.run_ear_tag_detection_stage(
        video_path=side_clip,
        session_id=config.session_id,
        cow_temp_id=cow_temp_id,
        output_dir=config.output_dir,
        view="side",
        show_preview=False,
    )
    print(f"[Stage 3] Completed! Ear-tag metadata: {metadata_path}")
    return metadata_path


def run_stage_last_row_detection(config: PipelineConfig, cow_temp_id: str, predictor: Predict) -> str:
    """Stage 4: Run last-row detection on ear-tag crops and enrich metadata."""
    saved_path = predictor.run_last_row_detection_stage(
        session_id=config.session_id,
        cow_temp_id=cow_temp_id,
        output_dir=config.output_dir,
        show_preview=True
    )
    print(f"[Stage 4] Last-row detection complete: {saved_path}")
    return saved_path


def run_stage_ocr(_: PipelineConfig, __: str) -> None:
    """Stage 5 placeholder: read crops and write OCR/Cow-ID metadata."""
    print("[Stage 5] TODO: implement OCR artifact writer")


def run_stage_pose_and_lameness(config: PipelineConfig, cow_temp_id: str) -> None:
    """Stage 6: Run pose estimation on top view and update lameness_analysis.json."""
    import cv2
    cow_folder = os.path.join(config.output_dir, config.session_id, cow_temp_id)
    top_dir = os.path.join(cow_folder, "top")
    # Find top video
    top_video = None
    if os.path.exists(top_dir):
        for f in os.listdir(top_dir):
            if f.endswith(".mp4"):
                top_video = os.path.join(top_dir, f)
                break
    if not top_video or not os.path.exists(top_video):
        print(f"[Stage 6] No top video found for {cow_temp_id}, skipping pose analysis.")
        return
    # Extract frames
    cap = cv2.VideoCapture(top_video)
    frames = []
    count = 0
    max_frames = 100  # Limit for speed; adjust as needed
    while True:
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        frames.append(frame)
        count += 1
    cap.release()
    if not frames:
        print(f"[Stage 6] No frames extracted from {top_video} for {cow_temp_id}.")
        return
    # Load pose model
    try:
        from ultralytics import YOLO
        pose_model = YOLO(config.pose_model_path)
    except Exception as e:
        print(f"[Stage 6] Failed to load pose model: {e}")
        return
    # Compute lameness score
    try:
        lameness_result = compute_lameness_score(frames, pose_model)
    except Exception as e:
        print(f"[Stage 6] Lameness scoring failed for {cow_temp_id}: {e}")
        return
    # Save to JSON
    create_or_update_cow_json(
        cow_id=cow_temp_id,
        session_id=config.session_id,
        video_path=top_video,
        cow_folder=cow_folder,
        **lameness_result
    )
    print(f"[Stage 6] Lameness analysis updated for {cow_temp_id}")


def run_stage_process_pose_and_lameness_for_cow(config: PipelineConfig, cow_temp_id: str) -> None:
    """Stage: Run pose estimation and lameness scoring for a single cow by delegating to detect_cows module."""
    from detect_cows import process_pose_and_lameness_for_cow
    from models.inference import PoseEstimation
    pose_estimator = PoseEstimation(config.pose_model_path)
    process_pose_and_lameness_for_cow(
        cow_temp_id=cow_temp_id,
        session_id=config.session_id,
        output_dir=config.output_dir,
        pose_estimator=pose_estimator,
        max_frames=100
    )


def run_stage_dashboard() -> None:
    """Stage 7: Launch web dashboard for human review and corrections."""
    subprocess.run("streamlit run review_dashboard.py", shell=True, check=False)


if __name__ == "__main__":
    # This is a skeleton entry point. Keep stages explicit and run only what is ready.
    config_ = PipelineConfig()
    predictor = Predict(
        tag_detection_model=config_.tag_model_path,
        row_detection_model=config_.row_model_path,
        pose_estimation_model=config_.pose_model_path,
        ocr_model="",
    )
    predictor.load_model()

    # Example full order:
    # run_stage_segmentation(config=config_)         # (1)
    # run_stage_extract_clips(config=config_)        # (2)

    # Run stages 3-6 for all cows in all sessions
    # processed_root = config_.output_dir
    # if os.path.exists(processed_root):
    #     for session in sorted(os.listdir(processed_root)):
    #         session_path = os.path.join(processed_root, session)
    #         if not os.path.isdir(session_path) or not session.startswith("session"):
    #             continue
    #         print(f"[Pipeline] Processing session: {session}")
    #         for cow in sorted(os.listdir(session_path)):
    #             cow_path = os.path.join(session_path, cow)
    #             if not os.path.isdir(cow_path) or not cow.startswith("cow_"):
    #                 continue
    #             print(f"[Pipeline] Processing cow: {cow}")
    #             try:
    #                 run_stage_ear_tag_metadata(config_, cow, predictor)
    #             except Exception as e:
    #                 print(f"[Pipeline] Ear tag metadata failed for {cow}: {e}")
    #             try:
    #                 run_stage_last_row_detection(config_, cow, predictor)
    #             except Exception as e:
    #                 print(f"[Pipeline] Last row detection failed for {cow}: {e}")
    #             try:
    #                 run_stage_pose_and_lameness(config_, cow)
    #             except Exception as e:
    #                 print(f"[Pipeline] Pose/lameness failed for {cow}: {e}")

    run_stage_dashboard()            # (7) launch dashboard for review
