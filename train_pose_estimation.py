from ultralytics import YOLO

# Path to your custom data.yaml
DATA_YAML = '../../EarTagModel/Pose/cattleeyeview_pose.yaml'  # Update this path if needed

# Path to the pre-trained YOLOv8 pose model
PRETRAINED_MODEL = '../../EarTagModel/Pose/yolov8n-pose.pt'  # Or yolov8s-pose.pt, etc.

# Output directory for training results
OUTPUT_DIR = 'runs/pose/cow_pose_finetune'

# Training parameters
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640

def main():
    # Load the pre-trained model
    model = YOLO(PRETRAINED_MODEL)

    # Train the model on your cow dataset
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch= BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=OUTPUT_DIR,
        device=0  # Set to 'cpu' if no GPU
    )

if __name__ == '__main__':
    main()