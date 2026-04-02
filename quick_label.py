import cv2
from ultralytics import YOLO
import os
import time
import utils.video as video_utils
from pipeline_contracts import (
    build_ear_tag_metadata,
    enrich_record_with_last_row,
    load_json_document,
    write_json_document,
)

class Predict:
    
    def __init__(self, tag_detection_model: str, row_detection_model: str, pose_estimation_model: str, ocr_model: str):
        
        self.tag_detection_model = tag_detection_model
        self.row_detection_model = row_detection_model
        self.pose_estimation_model = pose_estimation_model
        self.ocr_model = ocr_model
        self.tag_model = None
        self.row_model = None
        
    def run_ear_tag_detection_stage(
        self,
        video_path: str,
        session_id: str,
        cow_temp_id: str,
        output_dir: str = "data/processed",
        view: str = "side",
        show_preview: bool = False,
    ):
        """
        Run ear-tag detection on a cow clip and save canonical metadata.

        Returns:
            str: Path to saved ear_tag_metadata.json
        """
        if self.tag_model is None:
            raise ValueError("Tag model not loaded. Call load_model() first.")

        cap = self.load_video(video_path)
        frame_index = 0
        detection_records = []
        
        ear_tag_path = os.path.join(
            output_dir, 
            session_id,
            cow_temp_id,
            "ear_tag")

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                results = self.predict_tags(frame)
                if not results:
                    frame_index += 1
                    continue

                result = results[0]
                if len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()

                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        
                        #Grab ear tag crop and save it
                        ear_tag_crop = self.extract_ear_tag_crop(frame, box)
                        file_name = f"f{frame_index:06d}_d{i:02d}.jpg"
                        ear_tag_crop_path = os.path.join(ear_tag_path, file_name)
                        cv2.imwrite(ear_tag_crop_path, ear_tag_crop)
                        print(f"Saved Ear tag crop: {ear_tag_crop_path}")
                        
                        record_id = f"rec_f{frame_index:06d}_d{i:02d}"
                        detection_records.append(
                            {
                                "record_id": record_id,
                                "frame_index": frame_index,
                                "bbox_xyxy": [x1, y1, x2, y2],
                                "confidence": float(confidences[i]),
                                "class_id": int(class_ids[i]),
                            }
                        )

                if show_preview:
                    annotated = result.plot()
                    cv2.imshow("Ear Tag Detection", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()

        payload = build_ear_tag_metadata(
            session_id=session_id,
            cow_temp_id=cow_temp_id,
            source_video_path=video_path,
            records=detection_records,
            view=view,
        )

        metadata_path = os.path.join(
            ear_tag_path,
            "ear_tag_metadata.json"
        )
        
        saved_path = write_json_document(metadata_path, payload)
        print(f"Saved ear-tag metadata: {saved_path}")
        return str(saved_path)
    
    def extract_ear_tag_crop(self, frame, bbox):
        """Extract the ear tag region from the frame using the bounding box."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        ear_tag_crop = frame[y1:y2, x1:x2]
        
        return ear_tag_crop

    def run_last_row_detection_stage(
        self,
        session_id: str,
        cow_temp_id: str,
        output_dir: str = "data/processed",
        show_preview: bool = False,
    ):
        """
        Run last-row detection on saved ear-tag crops and enrich ear_tag_metadata.json.

        Reads each crop saved by run_ear_tag_detection_stage, runs the OBB row model,
        and calls enrich_record_with_last_row for every detection found.

        Returns:
            str: Path to the updated ear_tag_metadata.json
        """
        if self.row_model is None:
            raise ValueError("Row model not loaded. Call load_model() first.")

        ear_tag_dir = os.path.join(output_dir, session_id, cow_temp_id, "ear_tag")
        metadata_path = os.path.join(ear_tag_dir, "ear_tag_metadata.json")

        metadata_document = load_json_document(metadata_path)
        detection_records = metadata_document.get("records", [])

        if not detection_records:
            print("No ear-tag records found. Skipping last-row detection.")
            return metadata_path

        enriched_count = 0
        stop_preview = False
        try:
            for record in detection_records:
                record_id = record["record_id"]
                # record_id format: "rec_f000030_d00"  ->  crop: "f000030_d00.jpg"
                file_name = record_id[4:] + ".jpg"  # strip leading "rec_"
                ear_tag_crop_path = os.path.join(ear_tag_dir, file_name)

                if not os.path.exists(ear_tag_crop_path):
                    print(f"Crop not found, skipping: {ear_tag_crop_path}")
                    continue

                ear_tag_crop = cv2.imread(ear_tag_crop_path)
                if ear_tag_crop is None:
                    print(f"Could not read crop image, skipping: {ear_tag_crop_path}")
                    continue

                row_results = self.predict_last_row(ear_tag_crop)
                if not row_results:
                    continue
                
                

                row_result = row_results[0]
                if row_result.obb is None or len(row_result.obb) == 0:
                    if show_preview:
                        #resize ear_tag_crop for better visualization of OBB results
                        ear_tag_crop = cv2.resize(ear_tag_crop, (512, 512))
                        cv2.imshow("Last Row Detection", ear_tag_crop)
                        if cv2.waitKey(3000) & 0xFF == ord("q"):
                            stop_preview = True
                            break
                    continue

                row_obb = row_result.obb
                # Pick the highest-confidence detection from the OBB model
                best_idx = int(row_obb.conf.argmax())
                x1, y1, x2, y2 = [int(v) for v in row_obb.xyxy[best_idx].cpu().numpy()]
                row_confidence = float(row_obb.conf[best_idx].cpu().numpy())

                enrich_record_with_last_row(metadata_document, record_id, [x1, y1, x2, y2], row_confidence)
                enriched_count += 1

                if show_preview:
                    annotated_crop = row_result.plot()
                    annotated_crop = cv2.resize(annotated_crop, (512, 512))
                    cv2.imshow("Last Row Detection", annotated_crop)
                    if cv2.waitKey(3000) & 0xFF == ord("q"):
                        stop_preview = True
                        break
        finally:
            if show_preview:
                cv2.destroyAllWindows()

        saved_path = write_json_document(metadata_path, metadata_document)
        if stop_preview:
            print("Preview stopped by user (pressed 'q').")

        print(
            f"Last-row detection complete. Enriched {enriched_count}/{len(detection_records)} records. Saved: {saved_path}"
        )
        return str(saved_path)

    def run_pose_estimation_top_view(self, video_path: str):
        """Run pose estimation on the top view video and visualize keypoints in real time."""
        if not hasattr(self, 'pose_model') or self.pose_model is None:
            if os.path.exists(self.pose_estimation_model):
                self.pose_model = YOLO(self.pose_estimation_model)
                print(f"Loaded pose estimation model from {self.pose_estimation_model}")
            else:
                raise FileNotFoundError(f"Pose Estimation model not found at {self.pose_estimation_model}")

        cap = self.load_video(video_path)
        frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        seg = video_utils.Segmenter()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            #Slice frame
            frame = seg.slice(frame)
            
            # Run pose estimation
            results = self.pose_model(frame)
            annotated = results[0].plot() if results and hasattr(results[0], 'plot') else frame
            # Show the annotated frame
            # frame_resized = cv2.resize(annotated, (1280, 720))
            
            cv2.imshow('Pose Estimation - Top View', annotated)
            # If keypoints detected, delay for 3 seconds
            keypoints_detected = False
            if results and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                kps = results[0].keypoints.xy.cpu().numpy()
                if kps.shape[0] > 0:
                    pass
                    keypoints_detected = True
            if keypoints_detected:
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames for pose estimation.")
    
    
        
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
        
        if os.path.exists(self.pose_estimation_model):
            self.pose_model = YOLO(self.pose_estimation_model)
            print(f"Loaded pose estimation model from {self.pose_estimation_model}")
        else:
            raise FileNotFoundError(f"Pose Estimation model not found at {self.pose_estimation_model}")
        
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
            # Optional: display frame (comment out if not needed)
            frame_resized = cv2.resize(annotated, (1920,1080)) #Resize for display purposes
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
    pose_model_path = "runs/pose/runs/pose/cow_pose_finetune/train4/weights/best.pt"
    ocr_model_path = ""  # Placeholder for OCR model if needed
    
    predictor = Predict(tag_model_path, row_model_path, pose_model_path, ocr_model_path)
    predictor.load_model()
    
    # Assuming 4K video is in data/raw/side_4k/ or similar
    video_path = "data/raw/side_4k/side_4k.mp4"  # Replace with actual video path
    top_video_path = "data/raw/top/top.mp4" #Replace 
    
    # cam = cv2.VideoCapture(video_path)
    # total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(total_frames)
    predictor.run_pose_estimation_top_view(top_video_path)
    # Example stage writer call:
    # predictor.run_ear_tag_detection_stage(
    #     video_path=video_path,
    #     session_id="session_1",
    #     cow_temp_id="cow_0",
    #     show_preview=True,
    # )
        