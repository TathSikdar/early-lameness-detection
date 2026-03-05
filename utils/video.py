import cv2
import matplotlib.pyplot as plt
import time as tm
import numpy as np
import os
import shutil

TOP = 0
SIDE = 1
FRONT = 2

class Segmenter:
    def __init__(self, top_path='data/raw/top/top.mp4', side_path='data/raw/side/side.mp4', front_path='data/raw/front/front.mp4',
                 TOP_SLICE_DATA= (140, 640, 100, 300),
                 SIDE_SLICE_DATA= (200, 640, 150, 360),
                 FRONT_SLICE_DATA= (150, 420, 100, 350)):
        self.view_paths = {
            'top': (top_path, TOP),
            'side': (side_path, SIDE),
            'front': (front_path, FRONT)
        }
        self.TOP_X_START = TOP_SLICE_DATA[0]
        self.TOP_X_END = TOP_SLICE_DATA[1]
        self.TOP_Y_START = TOP_SLICE_DATA[2]
        self.TOP_Y_END = TOP_SLICE_DATA[3]
        self.SIDE_X_START = SIDE_SLICE_DATA[0]  
        self.SIDE_X_END = SIDE_SLICE_DATA[1]
        self.SIDE_Y_START = SIDE_SLICE_DATA[2]
        self.SIDE_Y_END = SIDE_SLICE_DATA[3]
        self.FRONT_X_START = FRONT_SLICE_DATA[0]
        self.FRONT_X_END = FRONT_SLICE_DATA[1]
        self.FRONT_Y_START = FRONT_SLICE_DATA[2]
        self.FRONT_Y_END = FRONT_SLICE_DATA[3]

    def create_cow_folders(self, output_dir, session_id, cow_id):
        """
        Creates the necessary subfolders for a cow's segmented data.
        
        Input:
            output_dir (str): Base output directory.
            session_id (str): Session identifier.
            cow_id (str): Cow identifier.
        
        Creates folders: output_dir/session_id/cow_id/front, /top, /side, /ear_tag
        """
        cow_dir = os.path.join(output_dir, session_id, cow_id)
        folders = ['front', 'top', 'side', 'ear_tag']
        
        for folder in folders:
            folder_path = os.path.join(cow_dir, folder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created folder: {folder_path}")
    
    def capture_background_frame(self, camera_view, frame_number, output_dir, session_id):
        """
        Captures and saves the background frame from the original video for background subtraction.
        
        Args:
            camera_view (str): 'top', 'side', or 'front'
            frame_number (int): Frame number to capture as background
            output_dir (str): Base output directory
            session_id (str): Session identifier
        
        Saves the sliced background frame as output_dir/session_id/background_{camera_view}.jpg
        """
        if camera_view not in self.view_paths:
            raise ValueError(f"Invalid camera_view: {camera_view}. Must be 'top', 'side', or 'front'")
        
        video_path, view_const = self.view_paths[camera_view]
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Could not read frame {frame_number} from {video_path}")
        
        cap.release()
        
        # Slice the frame
        sliced_frame = self.slice(frame, camera=view_const)
        
        # Save to session folder
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        background_path = os.path.join(session_dir, f'background_{camera_view}.jpg')
        cv2.imwrite(background_path, sliced_frame)
        
        print(f"Background frame saved to: {background_path}")
        return background_path
        
    def display_all(self, frame_delay=-1):
        frame_count = 0
        top_path, const = self.view_paths['top']
        side_path, const = self.view_paths['side']
        front_path, const = self.view_paths['front']
        
        front_cam = cv2.VideoCapture(front_path)
        side_cam = cv2.VideoCapture(side_path)
        top_cam = cv2.VideoCapture(top_path)
        
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
                ignore_frame_flag = False
                continue
            elif not ignore_frame_flag:
                ignore_frame_flag = True
                pass
            
            
            front_sliced = self.slice(front_frame, camera=FRONT)
            top_sliced = self.slice(top_frame, camera=TOP)
            side_sliced = self.slice(side_frame, camera=SIDE)
            
            #Only time delay if the user specifies a time delay, otherwise just run as fast as possible
            if(frame_delay > 0):
                tm.sleep(frame_delay)
            else:
                pass
            
            if not (ret1 and ret2 and ret3):
                raise ValueError("Error reading frames from one of the videos.")    
            
            #limit set for now so that video does not read indefinitely, can be removed later!
            if frame_count >= 3000:
                break
            
            
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
                return (frame[self.TOP_Y_START:self.TOP_Y_END,self.TOP_X_START:self.TOP_X_END,:])
            elif camera == FRONT:
                return (frame[self.FRONT_Y_START:self.FRONT_Y_END,self.FRONT_X_START:self.FRONT_X_END,:])
            elif camera == SIDE:
                return (frame[self.SIDE_Y_START:self.SIDE_Y_END,self.SIDE_X_START:self.SIDE_X_END,:])
        except:
            raise ValueError("Value error of frame received, cannot slice frame!")

class Cleanup:
    def __init__(self):
        pass
    
    def remove_cow_folders(self, output_dir, session_id, cow_id):
        """
        Removes the subfolders created for a cow's segmented data.
        
        Input:
            output_dir (str): Base output directory.
            session_id (str): Session identifier.
            cow_id (str): Cow identifier.
        
        Removes folders: output_dir/session_id/cow_id/front, /top, /side, /ear_tag
        """
        cow_dir = os.path.join(output_dir, session_id, cow_id)
        folders = ['front', 'top', 'side', 'ear_tag']
        
        for folder in folders:
            folder_path = os.path.join(cow_dir, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"Removed folder: {folder_path}")
            else:
                print(f"Folder does not exist: {folder_path}")
        
        # Optionally remove the cow_dir if empty
        try:
            os.rmdir(cow_dir)
            print(f"Removed empty cow directory: {cow_dir}")
        except OSError:
            print(f"Cow directory not empty or does not exist: {cow_dir}")
            
    
            
    
    def remove_all(self, output_dir, session_id, remove_session_folder=False):
        """
        Remove every cow folder under a session. Optionally delete the session directory itself.
        
        This implementation leverages ``remove_cow_folders`` to ensure
        consistent cleanup logic for each cow.
        
        Args:
            output_dir (str): Base output directory.
            session_id (str): Session identifier.
            remove_session_folder (bool): if True, delete the session folder after cleaning.
        """
        session_dir = os.path.join(output_dir, session_id)
        if not os.path.exists(session_dir):
            print(f"Session directory does not exist: {session_dir}")
            return
        
        # delete all cows inside using existing helper
        for item in os.listdir(session_dir):
            cow_dir = os.path.join(session_dir, item)
            if os.path.isdir(cow_dir):
                self.remove_cow_folders(output_dir, session_id, item)
        
        if remove_session_folder:
            try:
                shutil.rmtree(session_dir)
                print(f"Removed session directory: {session_dir}")
            except OSError:
                print(f"Could not remove session directory (not empty?): {session_dir}")
    
seg = Segmenter()
cleaner = Cleanup()


# seg.display_all(frame_delay=0.5)
seg.capture_background_frame(camera_view='top', frame_number=2000, output_dir="data/processed", session_id="session_1")
seg.create_cow_folders(output_dir="data/processed", session_id="session_1", cow_id="cow_1")
seg.create_cow_folders(output_dir="data/processed", session_id="session_1", cow_id="cow_2")
seg.create_cow_folders(output_dir="data/processed", session_id="session_1", cow_id="cow_3")

# cleaner.remove_cow_folders(output_dir="data/processed", session_id="session_1", cow_id="cow_1") 
cleaner.remove_all(output_dir="data/processed", session_id="session_1", remove_session_folder=True)
    