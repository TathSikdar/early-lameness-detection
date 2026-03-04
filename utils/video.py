#This python script will be used to segment the long video session into many smaller clips 7-9 sec
#for each cow. The structure will consist of 
import cv2
import matplotlib.pyplot as plt
import time as tm

TOP_X_START = 140
TOP_X_END = 640
TOP_Y_START = 100
TOP_Y_END = 300

SIDE_X_START = 200
SIDE_X_END = 640
SIDE_Y_START = 150
SIDE_Y_END = 360

FRONT_X_START = 150
FRONT_X_END = 420
FRONT_Y_START = 100
FRONT_Y_END = 350

TOP = 0
SIDE = 1
FRONT = 2

class Segmenter:
    def __init__(self):
        pass
    
    def display_vid(self, path_to_vid, view=TOP):
        """ 
        Purpose: This function displays the sliced video of the camera view requestion
        Input:
            path_to_vid: The path to the video to display in string
            view: TOP, SIDE, or FRONT constant for video to display
        Output:
            A window displaying video
        """
        cam = cv2.VideoCapture(path_to_vid)
        
        if not cam:
            print("Error: Could not open video.")
            
        while True:
            ret, frame = cam.read() #Read one frame
            
            if not ret:
                break
            
            frame = self.slice(frame, camera=view)
            cv2.imshow("Video", frame)  
            
            #Wait 1ms; break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()
        
    def display_all(self, front, side, top):
        frame_count = 0
        front_cam = cv2.VideoCapture(front)
        side_cam = cv2.VideoCapture(side)
        top_cam = cv2.VideoCapture(top)
        
        #Get the total numbers for frames from the metadata of the file
        front_count = int(front_cam.get(cv2.CAP_PROP_FRAME_COUNT))
        side_count = int(side_cam.get(cv2.CAP_PROP_FRAME_COUNT))
        top_count = int(top_cam.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Front Count: {front_count}, Side Count: {side_count}, Top Count: {top_count}")
        
        #Downscale the FPS by 50%
        ignore_frame_flag = False
        
        while True:
            frame_count += 1
            
            ret1, front_frame = front_cam.read()
            ret2, side_frame = side_cam.read()
            ret3, top_frame = top_cam.read()
            
            if ignore_frame_flag:
                #ignores the current frame
                print("Ignore frame flag is True, making false")
                ignore_frame_flag = False
                continue
            elif not ignore_frame_flag:
                print("NOT")
                ignore_frame_flag = True
                pass
            
            
            front_sliced = self.slice(front_frame, camera=FRONT)
            top_sliced = self.slice(top_frame, camera=TOP)
            side_sliced = self.slice(side_frame, camera=SIDE)
            
            tm.sleep(1)
            
            if not (ret1 and ret2 and ret3):
                print("Error: One of the frames not read")
                break
            
            if frame_count >= 3000:
                break
            
            current_frame_id = int(front_cam.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"Current frame ID: {current_frame_id}")
            
            #Resize frames
            width, height = 320, 240
            top_frame = cv2.resize(top_frame, (width, height))
            side_frame = cv2.resize(side_frame, (width, height))
            front_frame = cv2.resize(front_frame, (width, height))
            
            top_sliced = cv2.resize(top_sliced, (width,height))
            side_sliced = cv2.resize(side_sliced, (width,height))
            front_sliced = cv2.resize(front_sliced, (width, height))
            
            top_row = cv2.hconcat([side_frame, front_frame, top_frame])
            bottom_row = cv2.hconcat([side_sliced, front_sliced, top_sliced])
            
            combined = cv2.vconcat([top_row, bottom_row])
            
            
            cv2.imshow("Three Views", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Frame Count: {frame_count}")
                break
        
        
        
        front_cam.release()
        side_cam.release()
        top_cam.release()
        
        print(f"Frame Count: {frame_count}")
        
        
    def display_single_frame(self, path_to_vid, view=TOP):
        """
        Purpose: This function display a single frame of the request camera type with pixel value axis
            aids in figuring pixels value to slice frame for.
        Input:
            path_to_vid: The path to the video to display in string.
            view: TOP, SIDE, or FRONT constant for video to display.
        Output:
            A display window with frame 
        """
        cam = cv2.VideoCapture(path_to_vid)
        
        if not cam:
            print("Error: Could not open video")
            
        ret, frame = cam.read()
        if not ret:
            print("Error: Frame read failed")
            return
        
        h, w, c = frame.shape
        print(f"Height: {h}, Width: {w}, Channel: {c}")
        self.show(frame)
        
        cam.release()
        return
                    
            
    def show(self, frame):
        """
        Purpose: (HELPER) Internal function used to show the frame using plt
        Input:
            frame: a cv2 frame to display.
        Output:
            Display window with frame
        """
        plt.imshow(frame)
        plt.show()
        
    def slice(self, frame, camera=TOP):
        """
        Purpose: This function retuns the sliced frame based on input video data
        Input:
            frame: A cv2 frame to slice
            camera: TOP, FRONT, or SIDE constant to decide which pixel values to use when slicing
        Output:
            A sliced frame
        """
        try:
            if camera == TOP:
                return (frame[TOP_Y_START:TOP_Y_END,TOP_X_START:TOP_X_END,:])
            elif camera == FRONT:
                return (frame[FRONT_Y_START:FRONT_Y_END,FRONT_X_START:FRONT_X_END,:])
            elif camera == SIDE:
                return (frame[SIDE_Y_START:SIDE_Y_END,SIDE_X_START:SIDE_X_END,:])
        except:
            raise ValueError("Value error of frame received, cannot slice frame!")

        
    
seg = Segmenter()
side = "data/raw/side/side.mp4"
top = "data/raw/top/top.mp4"
front = "data/raw/front/front.mp4"

# seg.display_single_frame(path_to_vid, view=SIDE)
# seg.display_vid(path_to_vid=front, view=FRONT)
seg.display_all(front=front, side=side, top=top)
    
    
    
    