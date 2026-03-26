def pad_label_filenames_to_six_digits(pose_root):
	"""
	Renames label files in images/labels for train and val splits in the Pose dataset so that the frame number is always 6 digits (e.g., 02_000031.txt).
	This ensures label and image filenames match for YOLO.
	"""
	for split in ["train", "val"]:
		label_dir = os.path.join(pose_root, split, "labels")
		if not os.path.isdir(label_dir):
			continue
		for fname in os.listdir(label_dir):
			if not fname.endswith(".txt"):
				continue
			parts = fname.split("_")
			if len(parts) != 2:
				continue
			prefix, frame_part = parts
			frame_base, ext = os.path.splitext(frame_part)
			# Pad to 6 digits if not already
			if len(frame_base) < 6:
				new_frame = frame_base.zfill(6)
				new_name = f"{prefix}_{new_frame}{ext}"
				src = os.path.join(label_dir, fname)
				dst = os.path.join(label_dir, new_name)
				if not os.path.exists(dst):
					os.rename(src, dst)
					print(f"Renamed {src} -> {dst}")
				else:
					print(f"Skipping {src}, {dst} already exists")
import shutil

def flatten_and_rename_pose_data(pose_root):
	"""
	Flattens and renames files in images/labels for train and val splits in the Pose dataset.
	For each file in a source video folder, moves it up one level and renames it to sourceVideoFolder_frameNumber.ext.
	Deletes the source video folder if empty after moving.
	"""
	for split in ["train", "val"]:
		for data_type in ["images", "labels"]:
			base_dir = os.path.join(pose_root, split, data_type)
			if not os.path.isdir(base_dir):
				continue
			for src_video_folder in os.listdir(base_dir):
				src_path = os.path.join(base_dir, src_video_folder)
				if not os.path.isdir(src_path):
					continue
				for fname in os.listdir(src_path):
					src_file = os.path.join(src_path, fname)
					if not os.path.isfile(src_file):
						continue
					# Remove extension from folder name if present (e.g., 02.mp4 -> 02)
					folder_base = os.path.splitext(src_video_folder)[0]
					frame_base, ext = os.path.splitext(fname)
					new_name = f"{folder_base}_{frame_base}{ext}"
					dest_file = os.path.join(base_dir, new_name)
					# If file exists, skip to avoid overwrite
					if os.path.exists(dest_file):
						print(f"Skipping existing file: {dest_file}")
						continue
					shutil.move(src_file, dest_file)
					print(f"Moved {src_file} -> {dest_file}")
				# Remove folder if empty
				if not os.listdir(src_path):
					os.rmdir(src_path)
					print(f"Removed empty folder: {src_path}")

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
	# dataset_path = "C:/Users/fuzail_laptop/Downloads"
	pose_root = "/u50/fuzailm/EarTagModel/Pose"
	print(f"Running pad_label_filenames_to_six_digits on {pose_root}")
	# flatten_and_rename_pose_data(pose_root=pose_root)
	pad_label_filenames_to_six_digits(pose_root=pose_root)
