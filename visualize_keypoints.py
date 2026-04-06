
import cv2
import numpy as np
from models.inference import PoseEstimation


def _resize_for_display(image, max_width=1200, max_height=800):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        return cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image

def draw_pose_lines(image, keypoints, keypoint_radius=5, keypoint_color=(255, 0, 255)):
    # Keypoint indices for cow pose
    LEFT_EAR = 4
    RIGHT_EAR = 6
    WITHERS = 9
    TAIL_BASE = 22
    # Draw keypoints
    for idx, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:
            cv2.circle(image, (int(round(x)), int(round(y))), keypoint_radius, keypoint_color, -1)
    # Draw head bobbing line (left ear to right ear)
    if len(keypoints) > max(LEFT_EAR, RIGHT_EAR):
        pt1 = tuple(map(int, map(round, keypoints[LEFT_EAR])))
        pt2 = tuple(map(int, map(round, keypoints[RIGHT_EAR])))
        if all(pt1) and all(pt2):
            cv2.line(image, pt1, pt2, (0, 255, 255), 2)  # Yellow line
    # Draw spine alignment line (withers to tail base)
    if len(keypoints) > max(WITHERS, TAIL_BASE):
        pt1 = tuple(map(int, map(round, keypoints[WITHERS])))
        pt2 = tuple(map(int, map(round, keypoints[TAIL_BASE])))
        if all(pt1) and all(pt2):
            cv2.line(image, pt1, pt2, (0, 128, 255), 2)  # Orange line
    return image







def visualize_pose_video_inference(video_path, model_path, keypoint_radius=5, keypoint_color=(255, 0, 255), window_name="Pose Video Inference"):
    """Run pose inference on each frame of a video and visualize keypoints and lines live."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return
    pose_estimator = PoseEstimation(model_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = max(int(1000 / fps), 1) if fps and fps > 0 else 33
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            keypoints = None
            try:
                result = pose_estimator.model(frame)
                if result and hasattr(result[0], 'keypoints') and result[0].keypoints is not None:
                    kps = result[0].keypoints.xy.cpu().numpy()
                    if kps.shape[0] > 0:
                        keypoints = kps[0]
            except Exception as e:
                print(f"Pose inference failed: {e}")
            if keypoints is not None:
                frame = draw_pose_lines(frame, keypoints, keypoint_radius, keypoint_color)
            display_frame = _resize_for_display(frame)
            cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(0 if paused else frame_delay) & 0xFF
        if key in {ord("q"), 27}:
            break
        if key == ord(" "):
            paused = not paused
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage: update these paths as needed
    video_path = "data/raw/top/top_better_quality.mp4"  # Path to your test video
    model_path = "runs/pose/runs/pose/cow_pose_finetune/train6/weights/best.pt"  # Path to your pose model
    print(f"Video: {video_path}\nModel: {model_path}")
    visualize_pose_video_inference(video_path, model_path)
    