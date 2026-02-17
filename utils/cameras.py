# utils/cameras.py
import time
import matplotlib.pyplot as plt
import cv2
import os

class BaseCamera:
    def start(self):
        pass

    def read(self):
        """
        Returns:
            frame (np.ndarray) or None
            timestamp_ms (int) or None
        """
        #forces child class to implement necessary functionality
        raise NotImplementedError

    def stop(self):
        pass

#Goal is to output a single frame each time read() is run along with session 
class ImageFolderCamera(BaseCamera): #Frames sourced from a folder of images
    def __init__(self):
        self.index = 1
        self.working_path = os.getcwd()
        
    #Behaviour Undefined
    def start(self):
        self.last_time = time.time()

    def read(self, frame='curr'):
        if (frame.lower() == 'curr'):
            path = self.working_path
            path = path.replace('\\', '/')
            img_path = f"{path}/early-lameness-detection/data/raw/Ear_Tag/{self.index}.jpg"
            
            img = cv2.imread(img_path, 0)
            if img.any() == None:
                raise ValueError("Image not read, please fix path!")
            img = self.resize_img(img)
            
            return img
        
        elif (frame.lower() == 'next'):
            path = self.working_path
            path = path.replace('\\', '/')
            self.index += 1
            img_path = f"{path}/early-lameness-detection/data/raw/Ear_Tag/{self.index}.jpg"
            
            img = cv2.imread(img_path, 0)
            if img.any() == None:
                raise ValueError("Image not read, please fix path!")
            img = self.resize_img(img)
            
            return img
        else:
            raise ValueError("Inputted parameter for frame is incorrect, must either be \'curr\' or \'next\'")
        
            
    def resize_img(self, frame):
        #Resize all outgoing images to the same dimentions
        dst = cv2.resize(frame, dsize=(1280, 720), interpolation=cv2.INTER_LINEAR)
        
        return dst
    
    def reset(self):
        self.index = 1
        
    def stop(self):
        pass



# cam = ImageFolderCamera()
# frame = cam.read(frame='next')
# frame = cam.read(frame='next')
# plt.imshow(frame, cmap='gray')
# plt.show()