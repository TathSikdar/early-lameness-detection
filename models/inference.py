import cv2
import numpy as np
import re
import os
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from ultralytics import YOLO
import pandas as pd

#The Model has been trained on the CEID-D Dataset sourced from Kaggle which has a collection of images of cows with ear tags labeled.
#Current model is traind on the full color image, revisit this in the future and decide if training on grayscale images would be better.
#Date flow process:

#The frame first goes through the Ear tag detection model, this model will analyze the frame and output the relevan pixels
#Which it thinks the ear tag is located at. This data then needs to be fed into another model which will will analyze cropped images
#and output where it thinks the last row of the numbers is located. This is necessary becasue the OCR model needs a cropped image 
#which contains nothing other than the text. Hence another Yolo model is created to extract the location of the bottom most row.
class EarTagDectionAndLocaliztion:
    
    def __init__(self):
        #Add logic later
        pass
    
    def visualize_data(self, file_num):
        img_path = f"data/EarTagModel/cow_eartag_detection_dataset/train/images/cow{file_num}.jpg"
        lbl_path = f"data/EarTagModel/cow_eartag_detection_dataset/train/labels/cow{file_num}.txt"
        
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if frame.any() == None:
            raise FileNotFoundError("Image not found, please check path!")
        
        pix_hight, pix_wid = frame.shape
        
        with open(lbl_path, "r") as file:
            content = file.read()
            data = content.split()
            # print(content)
            
            lines = int(len(data)/5)
            
            # print(f"The Number of lines is {lines}")
            
            ear_tags = []
            
            for i in range(0, len(data), 5):
                #Get values from the list
                x_norm_center = float(data[i+1])
                y_norm_center = float(data[i+2])
                w_norm = float(data[i+3])
                h_norm = float(data[i+4])
                
                #Calculate coords of tags
                x_cent = int(pix_wid*x_norm_center)
                y_cent = int(pix_hight*y_norm_center)
                width = int(pix_wid*w_norm)
                hight = int(pix_hight*h_norm)
                                
                #Add all pictures of ear tags to a list
                ear_tags.append(frame[y_cent-hight:y_cent+hight,x_cent-width:x_cent+width]) 
                
                #Draw Circles around tags
                cv2.circle(frame, center=(x_cent,y_cent), radius=max(width,hight), color=(255,0,0), thickness=3)
                
            #Start the figure
            fig = plt.figure(figsize=(8, 6))
            num_eartags = lines            
            gs = GridSpec(
                2, num_eartags,
                height_ratios=[3, 1],   # big image taller than small ones
                hspace=0.3
            )
            ax_frame = fig.add_subplot(gs[0, :])
            ax_frame.imshow(frame, cmap="gray")
            ax_frame.set_title("Full image")
            ax_frame.axis("off")
            
            for i, tag in enumerate(ear_tags):
                ax = fig.add_subplot(gs[1, i])
                ax.imshow(tag, cmap="gray")
                ax.set_title(f"Tag {i+1}")
                ax.axis("off")

        plt.show()  
        file.close()   
        
    def clean_data(self):
        path = f"/u50/fuzailm/EarTagModel/cow_eartag_recognition_dataset/train/labels/cleaned_gt_eartags0.txt"
        
        try: 
            with open(path, "r+") as file:
                content = file.read()
                data = content.splitlines()
                data = data[0].split(",")
            file.close()
        except:
            pass
        
        length = len(data)
        print(f"Length is: {length}")
        print(0)
        for index in range(0, len(data)):
            if (index != (length -1)):
                print(data[index])
            else: 
                continue
            
    def clean_labels_for_last_row_detection(self):
        clean_train_or_val = "val"
        
        path = f"/u50/fuzailm/EarTagModel/cow_eartag_recognition_dataset/{clean_train_or_val}/labels"
        img_path = f"/u50/fuzailm/EarTagModel/cow_eartag_recognition_dataset/{clean_train_or_val}/images"
        
        # ++++++++++++++++++++ Process data and normalize data ++++++++++++++++
        # for i in range(2501,3238):
        #     try:
        #         label_path = path + f"/cleaned_gt_eartags{i}.txt"
        #         #Read in the images
        #         abs_img_path = img_path + f"/eartags{i}.jpg"
        #         img = cv2.imread(abs_img_path)
        #         img_height, img_width, channel = img.shape
        #         with open(label_path, "r+") as f:
        #             content = f.read()
        #             split_content = content.split(",")
        #             split_content.pop()
        #             to_write = "0"
        #             for val in range(0, len(split_content), 2):
        #                 #Nomalize the value before writing
        #                 normal_x = int(split_content[val]) / img_width
        #                 normal_y = int(split_content[val+1]) / img_height
                        
        #                 to_write += f",{normal_x},{normal_y}"
        #             to_write += '\n'
                    
        #             #Erase the old data and write the new one!
        #             f.seek(0)
        #             f.truncate()
        #             f.write(to_write)
        #         f.close()
        #     except Exception as e:
        #         #print(f"In Exception: {e} occured") 
        #         continue                    
        
        # ++++++++++++++++ Process data to remove commans and rewrite data +++++++++++
        # for i in range(2501, 3238):
        #     try:
        #         label_path = path + f"/cleaned_gt_eartags{i}.txt"
                
        #         #Open File
        #         with open(label_path, "r+") as f:
        #             content = f.read()
        #             content = content.replace(",", " ") #Replace the commands with spaces
                    
        #             #Move cursor back, remove old content, write new content
        #             f.seek(0)
        #             f.truncate()
        #             f.write(content)
        #         f.close()
        #     except:
        #         continue

        # ++++++++++++++++ Renaming the Python Files +++++++++++++
        for i in range(2501,3238):
            try:
                label_path = path + f"/cleaned_gt_eartags{i}.txt"
                os.rename(label_path, path + f"/eartags{i}.txt")
            except:
                pass
        
    def train_model(self, model):
        ''' 
        Input: Model is an integer value to choose which model to train:
        1 - Ear Tag detection
        0 - OCR Model
        '''
        import torch
        from ultralytics import YOLO
        
        epochs = 50
        
        #If model is 1, train ear tag detection model
        if(model):
            model = YOLO("/u50/fuzailm/EarTagModel/cow_eartag_detection_dataset/yolo26m.pt")
            if(torch.cuda.is_available()):
                model.train(data="/u50/fuzailm/EarTagModel/cow_eartag_detection_dataset/dataset_custom.yaml",project="Ear_Tag_Detection_Model",name=f"Epochs_{epochs}", imgsz=640, batch=8, epochs=epochs, workers=1, device=0)
            else:
                model.train(data="/u50/fuzailm/EarTagModel/cow_eartag_detection_dataset/dataset_custom.yaml",project="Ear_Tag_Detection_Model",name=f"Epochs_{epochs}", imgsz=640, batch=8, epochs=epochs, workers=1, device='cpu')
        else: #Otherwise train last row detection model
            model = YOLO("/u50/fuzailm/EarTagModel/cow_eartag_recognition_dataset/yolo26m-obb.pt")
            if(torch.cuda.is_available()):
                model.train(data="/u50/fuzailm/EarTagModel/cow_eartag_recognition_dataset/dataset_custom.yaml",project="Last_Row_Detection_Model",name=f"Epochs_{epochs}", imgsz=640, batch=8, epochs=epochs, workers=1, device=0)
            else:
                model.train(data="/u50/fuzailm/EarTagModel/cow_eartag_recognition_dataset/dataset_custom.yaml",project="Last_Row_Detection_Model",name=f"Epochs_{epochs}", imgsz=640, batch=8, epochs=epochs, workers=1, device='cpu')
        
    def predict_ear_tag_detection(self, path):
        from ultralytics import YOLO
        import cv2
        
        detection_mod = YOLO("/u50/fuzailm/cow_gait_analysis/early-lameness-detection/runs/detect/Detection_Model/Epochs_1/weights/best.pt")
        
        #Read one image and run inference on it
        frame = cv2.imread(path)
        result = detection_mod.predict(frame, show=True, save=True, name=f"predicted")
        
        return result
        
    
    def extract_earTags(self, result):
        #Currently this function simply extracts the images of the ear tags from the orignal image
        #After detection is run. Change functionality in the future to maybe write file to a path and then 
        #read the file for extracting numbers from ear tags. 
        for attr in result:
            
            frame = attr.orig_img
            
            for box in attr.boxes.xyxy:
                start_x, start_y, end_x, end_y = box
                start_x = int(start_x)
                start_y = int(start_y)
                end_x = int(end_x)
                end_y = int(end_y)
                
                ear_tag = frame[start_y:end_y, start_x:end_x]
                while 1:
                    cv2.imshow("Ear Tag", ear_tag)
                    if(cv2.waitKey(0) & 0xFF == 27):
                        break
                    
                cv2.destroyAllWindows()
                
class EarTagIDExtraction:
    
    def __init__ (self):
        #Add functionality later on as needed.
        pass

    def visualize_data(self, file_num):
        img_path = f"data/EarTagModel/cow_eartag_recognition_dataset/image/eartags{file_num}.jpg"
        # lbl_path = f"data/EarTagModel/cow_eartag_recognition_dataset/cleaned_labels/cleaned_gt_eartags{file_num}.txt"
        lbl_path = f"data/EarTagModel/cow_eartag_recognition_dataset/labels/gt_eartags{file_num}.txt"
        
        
        frame = cv2.imread(img_path)
        
        if frame.any() == None:
            raise FileNotFoundError("Image not found, please check path!")
        
        
        with open(lbl_path, "r") as file:
            content = file.read()
            data = re.split(r"[,\n]", content)
            
            data.pop() #Drop the last element because format of file includes empty line.
            for i in range(0, len(data), 9):
                #Get values from the list
                x1 = int(data[i])
                y1 = int(data[i+1])
                
                x2 = int(data[i+2])
                y2 = int(data[i+3])
                
                x3 = int(data[i+4])
                y3 = int(data[i+5])
                
                x4 = int(data[i+6])
                y4 = int(data[i+7])
                
                # extracted_num = int(data[i+8])
                                
                pts = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.int32)
                pts = pts.reshape((-1,1,2))
                
                #Draw Polylines around tags
                cv2.polylines(frame, [pts], True, (255,0,0), 1)
                
        plt.imshow(frame)            
        plt.show()  
        file.close()
        
    def clean_data(self):
        #This method removes the all but the last line from the labeled txt files and saved it under the cleaned_labels folder
        
        for file_num in range(0, 3238):
            
            lbl_path = f"data/EarTagModel/cow_eartag_recognition_dataset/labels/gt_eartags{file_num}.txt"
            cleaned_lbl_path = f"data/EarTagModel/cow_eartag_recognition_dataset/cleaned_labels/cleaned_gt_eartags{file_num}.txt"

            content = None
            data = None
            try:
                with open(lbl_path, "r") as file:
                    content = file.read()
                    data = content.splitlines()
                file.close()
            except:
                continue
            
            with open(cleaned_lbl_path, "w") as file:
                if(len(data) != 0):
                    file.write(f"{data[-1]}\n")
                else:
                    continue
            file.close()            
            
    def flag_wrong_format(self, start, end):
        #This method will show which bounding box was kept during the cleaning process. 
        #The user can either press <
        format_to_change = []
        
        for file_num in range(start, end):
            try:
                img_path = f"data/EarTagModel/cow_eartag_recognition_dataset/image/eartags{file_num}.jpg"
                lbl_path = f"data/EarTagModel/cow_eartag_recognition_dataset/cleaned_labels/cleaned_gt_eartags{file_num}.txt"
        
                frame = cv2.imread(img_path)
            
                if frame.any() == None:
                    continue
                else:
                    pass
        
                #Open each file and see draw the bounding box
                with open(lbl_path, "r") as file:
                    content = file.read()
                    data = re.split(r"[,\n]", content)
                
                    data.pop() #Drop the last element because format of file includes empty line.
                    for i in range(0, len(data), 9):
                        #Get values from the list
                        x1 = int(data[i])
                        y1 = int(data[i+1])
                    
                        x2 = int(data[i+2])
                        y2 = int(data[i+3])
                    
                        x3 = int(data[i+4])
                        y3 = int(data[i+5])
                    
                        x4 = int(data[i+6])
                        y4 = int(data[i+7])
                    
                        extracted_num = str(data[i+8])
                                    
                        pts = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.int32)
                        pts = pts.reshape((-1,1,2))
                    
                        #Draw Polylines around tags
                        cv2.polylines(frame, [pts], True, (0,0,255), 1)
                    
                file.close()
            
                while(True):
                    cv2.namedWindow(str(file_num), cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(str(file_num), 550, 300)
                    cv2.imshow(str(file_num), frame)
                    
                    if (cv2.waitKey(0) & 0xFF == ord('q')): #Press q for good.
                        break
                    elif (cv2.waitKey(0) & 0xFF == ord('p')): #Press p x2 for bad, will save file number to clean manually. 
                        format_to_change.append(str(file_num))
                        print(format_to_change)
                        break
                    elif (cv2.waitKey(0) & 0xFF == 27): #Press ESC x3 to quit.
                        #In the case that need to exit gracefully, write all appended changes thus far
                        with open("values_to_change.txt", "w") as file:
                            for val in format_to_change:
                                file.write(f"{str(val)} \n")
                            file.close()
                            
                        os._exit(1)
                    
                cv2.destroyAllWindows()
                
                with open("values_to_change.txt", "w") as file:
                    for val in format_to_change:
                        file.write(f"{str(val)} \n")
                
                file.close()
            except:
                continue
            
    def easy_ocr(self):
        import easyocr 
        print("In Easy OCR Function")
        #Creating an easyocr object and choosing language as english
        reader = easyocr.Reader(['en'])
        
        results = reader.readtext("data/EarTagModel/cow_eartag_recognition_dataset/image/eartags1.jpg")
        print(results)
        for (bbox, text, prob) in results:
            print(f"=============== In Function and Loop =============")
            print(f"Detected Text: {text}")
        
        
    def paddle_ocr(self):
        from paddleocr import PaddleOCR
                
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        img_path = "data/EarTagModel/cow_eartag_recognition_dataset/image/eartags1.jpg"
        
        #Read Image
        img = cv2.imread(img_path)
        
        plt.figure()
        plt.imshow(img)
        plt.show()
        result = ocr.ocr(img, cls=True)
        print(result[0][1][1])
        
        
        
        
#================== Testing ========================       
# Model 1: Ear Tag Detection
# detection_model = EarTagDectionAndLocaliztion()
# result = detection_model.clean_labels_for_last_row_detection()
# detection_model.train_model(0)

# #Model 2: OCR ID # extraction
# start = 3000
# end = 3300
# ocr = EarTagIDExtraction()
# # for i in range(0, 3238):
# # ocr.visualize_data(93)
# # ocr.clean_data()
# ocr.flag_wrong_format(start, end)
# # ocr.easy_ocr()
# # ocr.paddle_ocr()


class PoseEstimation:
    def __init__(self, model_path='runs/pose/runs/pose/cow_pose_finetune/train4/weights/best.pt'):
        """
        Initialize the pose estimation model.
        For cows, we may need a custom model, but starting with general pose.
        """
        self.model = None
        self.model_path = model_path

        if not os.path.exists(model_path):
            print(f"WARNING: pose model not found at {model_path}. Please download it or set the correct path.")
            return

        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"ERROR: failed to load YOLO model from {model_path}: {e}")
            self.model = None
    
    def estimate_pose(self, image_path, save_prediction=False):
        """
        Estimate pose from a single image.
        If save_prediction=True, saves the image with predicted keypoints to the same folder with '_predicted' suffix.
        Returns keypoints as numpy array.
        """
        if self.model is None:
            raise RuntimeError('Pose model not loaded.')

        results = self.model(image_path)
        if not results or len(results) == 0 or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            print(f"[WARN] No pose detected in image: {image_path}")
            return None
        keypoints = results[0].keypoints.xy.cpu().numpy()
        if keypoints.shape[0] == 0:
            print(f"[WARN] Detected object, but no keypoints found in: {image_path}")
            return None
        keypoints = keypoints[0]  # (N, 2)

        if save_prediction:
            import cv2
            import os
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERROR] Could not read image: {image_path}")
            else:
                for (x, y) in keypoints:
                    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                base, ext = os.path.splitext(image_path)
                save_path = base + "_predicted" + ext
                cv2.imwrite(save_path, img)
                print(f"Saved keypoints visualization to {save_path}")

        return keypoints
    
    def estimate_pose_video(self, video_path, output_csv=None):
        """
        Estimate pose for each frame in video and save to CSV.
        Format: video_name,frame,x1,y1,x2,y2,... for each keypoint.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        
        frame_count = 0
        keypoints_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame temporarily for inference
            temp_img = f"temp_frame_{frame_count}.jpg"
            cv2.imwrite(temp_img, frame)
            
            keypoints = self.estimate_pose(temp_img)
            if keypoints is not None:
                # Flatten keypoints
                kp_flat = keypoints.flatten()
                row = [os.path.basename(video_path), frame_count] + kp_flat.tolist()
                keypoints_list.append(row)
            
            os.remove(temp_img)
            frame_count += 1
        
        cap.release()
        
        if output_csv:
            df = pd.DataFrame(keypoints_list)
            df.to_csv(output_csv, index=False, header=False)
        
        return keypoints_list
    
    def display_keypoints(self, image_path, keypoints, radius=5, color=(0, 255, 0), thickness=-1):
        """
        Display the image with keypoints drawn on it.
        Args:
            image_path (str): Path to the image file.
            keypoints (np.ndarray): Array of shape (N, 2) with (x, y) keypoints.
            radius (int): Radius of keypoint circles.
            color (tuple): BGR color for keypoints.
            thickness (int): Thickness of keypoint circles (-1 for filled).
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return
        for (x, y) in keypoints:
            cv2.circle(img, (int(x), int(y)), radius, color, thickness)
        cv2.imshow("Keypoints", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_keypoints(self, image_path, keypoints, save_path, radius=5, color=(0, 0, 255), thickness=-1):
        """
        Save the image with keypoints drawn on it to disk.
        Args:
        image_path (str): Path to the image file.
            keypoints (np.ndarray): Array of shape (N, 2) with (x, y) keypoints.
            save_path (str): Path to save the output image.
            radius (int): Radius of keypoint circles.
            color (tuple): BGR color for keypoints.
            thickness (int): Thickness of keypoint circles (-1 for filled).
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return
        for (x, y) in keypoints:
            cv2.circle(img, (int(x), int(y)), radius, color, thickness)
        cv2.imwrite(save_path, img)
        print(f"Saved keypoints visualization to {save_path}")
        
if __name__ == "__main__":
    pose = PoseEstimation()
    vid_path = "C:/Users/fuzail_laptop/Desktop/Code/cow_gait_analysis/early-lameness-detection/data/raw/top/top.mp4"
    pose.estimate_pose_video(video_path=vid_path)
    