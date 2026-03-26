import cv2
import os

def visualize_keypoints(image_path, label_path, keypoint_radius=3, keypoint_color=(0, 0, 255)):
    """
    Visualizes keypoints on an image using YOLO-format label file.
    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the label file (YOLO keypoint format).
        keypoint_radius (int): Radius of the keypoint circle.
        keypoint_color (tuple): BGR color for keypoints.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    if not os.path.exists(label_path):
        print(f"Label not found: {label_path}")
        return

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            # YOLO keypoint format: class x1 y1 v1 x2 y2 v2 ...
            # We'll skip class and v (visibility) for now
            keypoints = parts[1:]
            for i in range(0, len(keypoints), 3):
                if i+2 >= len(keypoints):
                    break
                x = float(keypoints[i]) * w
                y = float(keypoints[i+1]) * h
                v = float(keypoints[i+2])
                if int(v) > 0:
                    cv2.circle(image, (int(x), int(y)), keypoint_radius, keypoint_color, -1)

    cv2.imshow('Keypoints', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = "../../EarTagModel/Pose/train/images/02_000031.jpg"
    lbl_path = "../../EarTagModel/Pose/train/labels/02_000031.txt"
    
    visualize_keypoints(image_path=img_path, label_path=lbl_path)