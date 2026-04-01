import cv2
from pathlib import Path


DEFAULT_POSE_ROOT = Path("C:/Users/fuzail_laptop/Downloads/Pose")


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


def _draw_yolo_pose_keypoints(image, label_path, keypoint_radius=3, keypoint_color=(255, 0, 255)):
    label_file = Path(label_path)
    if not label_file.exists():
        raise FileNotFoundError(f"Label not found: {label_file}")

    h, w = image.shape[:2]
    with label_file.open("r", encoding="utf-8") as label_handle:
        for line in label_handle:
            parts = line.strip().split()
            if len(parts) <= 5:
                continue

            keypoints = parts[5:]
            for index in range(0, len(keypoints), 3):
                if index + 2 >= len(keypoints):
                    break

                x = float(keypoints[index]) * w
                y = float(keypoints[index + 1]) * h
                visibility = float(keypoints[index + 2])
                if int(visibility) > 0:
                    cv2.circle(image, (int(round(x)), int(round(y))), keypoint_radius, keypoint_color, -1)

    return image


def _get_pose_frame_paths(pose_root, split, video_name, frame_number):
    pose_root_path = Path(pose_root)
    image_path = pose_root_path / split / "images" / video_name / f"{frame_number:06d}.jpg"
    label_path = pose_root_path / split / "labels" / video_name / f"{frame_number:05d}.txt"
    return image_path, label_path


def visualize_keypoints(image_path, label_path, keypoint_radius=5, keypoint_color=(255, 0, 255)):
    """Load an image, overlay YOLO pose keypoints, and display the result."""
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Image not found: {image_file}")
        return

    image = cv2.imread(str(image_file))
    if image is None:
        print(f"Could not read image: {image_file}")
        return

    image = _draw_yolo_pose_keypoints(
        image=image,
        label_path=label_path,
        keypoint_radius=keypoint_radius,
        keypoint_color=keypoint_color,
    )
    image = _resize_for_display(image)

    cv2.imshow("Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_video_frame_keypoints(
    source_video_path,
    split,
    frame_number,
    pose_root=DEFAULT_POSE_ROOT,
    keypoint_radius=5,
    keypoint_color=(255, 0, 255),
):
    """
    Read a frame from a source video, verify the saved Pose frame exists, then
    overlay the matching keypoints from Pose/<split>/labels/<video>/<frame>.txt.
    """
    if split not in {"train", "val"}:
        raise ValueError("split must be either 'train' or 'val'")

    video_file = Path(source_video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Source video not found: {video_file}")

    video_name = video_file.name
    
    image_path, label_path = _get_pose_frame_paths(pose_root, split, video_name, frame_number)

    if not image_path.exists():
        print(f"Saved Pose frame not found: {image_path}")
        return None

    if not label_path.exists():
        print(f"Pose label not found: {label_path}")
        return None

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_file}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
    finally:
        capture.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_file}")

    frame = _draw_yolo_pose_keypoints(
        image=frame,
        label_path=label_path,
        keypoint_radius=keypoint_radius,
        keypoint_color=keypoint_color,
    )

    cv2.putText(
        frame,
        f"{video_name} | {split} | frame {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    frame = _resize_for_display(frame)
    cv2.imshow("Video Frame Keypoints", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return frame


def visualize_video_keypoints_live(
    source_video_path,
    split,
    pose_root=DEFAULT_POSE_ROOT,
    keypoint_radius=5,
    keypoint_color=(255, 0, 255),
    window_name="Live Video Keypoints",
):
    """Play a source video and overlay keypoints on frames that exist in the Pose dataset."""
    if split not in {"train", "val"}:
        raise ValueError("split must be either 'train' or 'val'")

    video_file = Path(source_video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Source video not found: {video_file}")

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {video_file}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_delay = max(int(1000 / fps), 1) if fps and fps > 0 else 33
    frame_number = 0
    paused = False
    video_name = video_file.name

    try:
        while True:
            if not paused:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                image_path, label_path = _get_pose_frame_paths(pose_root, split, video_name, (frame_number+1))
                if image_path.exists() and label_path.exists():
                    frame = _draw_yolo_pose_keypoints(
                        image=frame,
                        label_path=label_path,
                        keypoint_radius=keypoint_radius,
                        keypoint_color=keypoint_color,
                    )
                    status_text = "keypoints"
                else:
                    status_text = "no label"

                cv2.putText(
                    frame,
                    f"{video_name} | {split} | frame {frame_number} | {status_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                display_frame = _resize_for_display(frame)
                cv2.imshow(window_name, display_frame)
                frame_number += 1

            key = cv2.waitKey(0 if paused else frame_delay) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == ord(" "):
                paused = not paused
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # visualize_video_keypoints_live(
    #     source_video_path="C:/Users/fuzail_laptop/Downloads/videos/06.mp4",
    #     split="train",
    # )
    
    # img_path, lbl_path = _get_pose_frame_paths(
    #     pose_root=DEFAULT_POSE_ROOT,
    #     split="train",
    #     video_name="06.mp4",
    #     frame_number=100
    # )
    
    
    img_path = "C:/Users/fuzail_laptop/Downloads/Pose/train/images/02.mp4/002323.jpg"
    lbl_path = "C:/Users/fuzail_laptop/Downloads/Pose/train/labels/02.mp4/02324.txt"
    
    print(img_path)
    print(lbl_path)
    
    visualize_keypoints(img_path,lbl_path)
    