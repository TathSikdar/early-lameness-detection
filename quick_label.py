import cv2
from ultralytics import YOLO
import os
import time

class Predict:
    
    def __init__(self, tag_detection_model: str, row_detection_model: str, ocr_model: str):
        
        self.tag_detection_model = tag_detection_model
        self.row_detection_model = row_detection_model
        self.ocr_model = ocr_model
        self.tag_model = None
        self.row_model = None
        
    def load_model(self):
        """Load the YOLO models for ear tag detection and last row detection."""
        if os.path.exists(self.tag_detection_model):
            self.tag_model = YOLO(self.tag_detection_model)
            print(f"Loaded tag detection model from {self.tag_detection_model}")
        else:
            raise FileNotFoundError(f"Tag detection model not found at {self.tag_detection_model}")
        
        if os.path.exists(self.row_detection_model):
            self.row_model = YOLO(self.row_detection_model)
            print(f"Loaded row detection model from {self.row_detection_model}")
        else:
            raise FileNotFoundError(f"Row detection model not found at {self.row_detection_model}")
        
    def load_video(self, video_path: str):
        """Load a video file and return the VideoCapture object."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file {video_path}")
        print(f"Loaded video from {video_path}")
        return cap
        
    def slice_frame(self, frame):
        """Placeholder for slicing frame if needed."""
        # Implement frame slicing logic if required
        pass
        
    def run_prediction_pipeline(self, video_path: str):
        """Run the full prediction pipeline on the video."""
        if self.tag_model is None or self.row_model is None:
            raise ValueError("Models not loaded. Call load_model() first.")
        
        cap = self.load_video(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 5616)
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            # Run predictions
            tag_results = self.predict_tags(frame)
            # row_results = self.predict_last_row(frame)
            
            # Here you can add logic to process results, e.g., draw on frame, save, etc.
            # For now, just print results
            print("----------------------------------------------------------")
            # print(f"Tag detection results: {tag_results}")
            # print(f"Row detection results: {row_results[0].orig_img}")
            print("----------------------------------------------------------")
            
            annotated = tag_results[0].plot() # Get the annotated image from the results
            
            if(len(tag_results[0].boxes) != 0):
                x1,y1,x2,y2 = tag_results[0].boxes.xyxy[0] 
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                
                ear_tag_crop = frame[y1:y2, x1:x2]
                
                row_results = self.predict_last_row(ear_tag_crop)
                
                predicted_last_row = row_results[0].plot() # Annotate the ear tag crop with row detection results    
                
                cv2.imshow("Ear Tag Crop", predicted_last_row)
                time.sleep(0.5)
            # Optional: display frame (comment out if not needed)
            frame_resized = cv2.resize(annotated, (1280,720)) #Resize for display purposes
            cv2.imshow('Frame', frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames.")
    
    def predict_tags(self, frame):
        """Run ear tag detection on a frame."""
        if self.tag_model is None:
            raise ValueError("Tag model not loaded.")
        results = self.tag_model(frame)
        return results
    
    def predict_last_row(self, frame):
        """Run last row detection on a frame."""
        if self.row_model is None:
            raise ValueError("Row model not loaded.")
        results = self.row_model(frame)
        return results
    
    def predict_cow_id(self, frame):
        """Placeholder for predicting cow ID using OCR."""
        # Implement OCR logic here if needed
        pass

if __name__ == "__main__":
    # Example usage
    tag_model_path = "runs/detect/Ear_Tag_Detection_Model/Epochs_50/weights/best.pt"
    row_model_path = "runs/obb/Last_Row_Detection_Model/Epochs_50/weights/best.pt"
    ocr_model_path = ""  # Placeholder for OCR model if needed
    
    predictor = Predict(tag_model_path, row_model_path, ocr_model_path)
    predictor.load_model()
    
    # Assuming 4K video is in data/raw/side_4k/ or similar
    video_path = "data/raw/side_4k/side_4k.mp4"  # Replace with actual video path
    
    # cam = cv2.VideoCapture(video_path)
    # total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(total_frames)
    predictor.run_prediction_pipeline(video_path)
        