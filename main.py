from dataclasses import dataclass
import os
import subprocess

from extract_frames import Extracter
from models.architecture import AnomalyDetector
from models.inference import PoseEstimation
from quick_label import Predict
from utils.metrics import GaitData
from utils.video import Segmenter


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
    """Stage 2: Read metadata and extract per-cow view clips."""
    extracter = Extracter(output_dir=config.output_dir, session_id=config.session_id)
    stats = extracter.extract_all_segments()
    print(f"[Stage 2] Extraction summary: {stats}")


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
    """Stage 6 placeholder: run pose inference, then calculate lameness."""
    top_clip = os.path.join(
        config.output_dir,
        config.session_id,
        cow_temp_id,
        "top",
        f"{cow_temp_id}_top_segment.mp4",
    )
    keypoints_csv = os.path.join(config.output_dir, config.session_id, cow_temp_id, "keypoints.csv")

    pose_estimator = PoseEstimation(model_path=config.pose_model_path)
    if pose_estimator.model is None:
        raise RuntimeError("Pose model could not be loaded")

    pose_estimator.estimate_pose_video(top_clip, keypoints_csv)
    gait = GaitData(keypoints_csv)
    features = gait.extract_features()

    detector = AnomalyDetector()
    print(f"[Stage 6] TODO: load trained anomaly model and score features: {features}")


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
    run_stage_ear_tag_metadata(config=config_, cow_temp_id="cow_0", predictor=predictor)     # (3) per cow
    run_stage_last_row_detection(config=config_, cow_temp_id="cow_0", predictor=predictor)  # (4) per cow
    # run_stage_ocr()                  # (5) per cow
    # run_stage_pose_and_lameness()    # (6) per cow
    # run_stage_dashboard()            # (7)
