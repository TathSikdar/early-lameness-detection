# Extract labeled frames from videos 01.mp4 to 14.mp4, saving to images/XX.mp4/ for each video
import os
import cv2

def extract_labeled_frames(dataset_path):
	videos_path = os.path.join(dataset_path, "videos")
	labels_path = os.path.join(dataset_path, "Pose/train/labels")
	images_path = os.path.join(dataset_path, "Pose/train/images")
	os.makedirs(images_path, exist_ok=True)

	# Iterate over all label folders (e.g., 01, 03, ...)
	for label_folder_name in os.listdir(labels_path):
		label_folder = os.path.join(labels_path, label_folder_name)
		if not os.path.isdir(label_folder):
			continue

		# Video file is e.g. 01.mp4
		video_file = f"{label_folder_name}"
		video_path = os.path.join(videos_path, video_file)
		if not os.path.isfile(video_path):
			print(f"Video file not found: {video_path}, skipping.")
			continue

		out_img_dir = os.path.join(images_path, label_folder_name)
		os.makedirs(out_img_dir, exist_ok=True)

		label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
		frame_indices = sorted([int(os.path.splitext(f)[0]) for f in label_files])

		if not frame_indices:
			print(f"No label files in {label_folder}, skipping.")
			continue

		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			print(f"Could not open video {video_path}")
			continue

		for frame_idx in frame_indices:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
			ret, frame = cap.read()
			if not ret:
				print(f"Could not read frame {frame_idx} from {video_path}")
				continue
			out_img_path = os.path.join(out_img_dir, f"{frame_idx:06d}.jpg")
			cv2.imwrite(out_img_path, frame)
			print(f"Saved {out_img_path}")

		cap.release()

if __name__ == "__main__":
	# Set your dataset path here (should contain videos/, labels/, images/)
	dataset_path = "C:/Users/fuzail_laptop/Downloads"
	extract_labeled_frames(dataset_path)
