import cv2
import matplotlib.pyplot as plt
import time as tm
import os
import json

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
        
    def background_subtract(self, background_path, frame):
        """
        Performs background subtraction on a given frame using a background image.
        
        Args:
            background_path (str): Path to the background image file.
            frame (numpy.ndarray): The current frame to subtract background from.
        
        Returns:
            numpy.ndarray: The frame with background subtracted (difference image).
        
        Raises:
            ValueError: If the background image cannot be loaded or dimensions don't match.
        """
        # Load the background image
        background = cv2.imread(background_path)
        if background is None:
            raise ValueError(f"Could not load background image from: {background_path}")
        
        # Check if dimensions match
        if background.shape != frame.shape:
            raise ValueError(f"Background image dimensions {background.shape} do not match frame dimensions {frame.shape}")
        
        # Perform background subtraction using absolute difference
        subtracted_frame = cv2.absdiff(frame, background)
        
        return subtracted_frame
        
    def create_base_metadata(self, session_id):
        """
        Creates the base metadata structure for a session.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            dict: Base metadata structure with session info and empty cow segments
        """
        metadata = {
            'session_id': session_id,
            'timestamp': tm.strftime('%Y-%m-%d %H:%M:%S'),
            'segments': {}
        }
        return metadata
    
    def aggregate_segments(self, output_dir, session_id):
        """
        Extracts segments from all three camera views and matches them by temporal alignment.
        Ensures that only compatible flag combinations are paired.
        
        Allowed front/top combinations:
        - GOOD + GOOD
        - GOOD + TOO_LONG
        - TOO_LONG + GOOD
        - TOO_LONG + TOO_LONG
        
        NOT allowed (segments are skipped):
        - Any combination with TOO_SHORT
        
        Args:
            output_dir (str): Base output directory
            session_id (str): Session identifier
            
        Returns:
            dict: Aggregated segment data with structure:
                  {
                    'cow_0': {'top': {...}, 'front': {...}, 'side': {...}},
                    'cow_1': {'top': {...}, 'front': {...}, 'side': {...}},
                    ...
                  }
        """
        print(f"\n{'='*60}")
        print("MATCHING SEGMENTS FROM TOP AND FRONT VIEWS")
        print(f"{'='*60}")
        
        # Extract all segments from front and top views
        print("\nExtracting segments from FRONT view...")
        front_segments_raw = self.extract_segments('front', output_dir, session_id)
        
        print("\nExtracting segments from TOP view...")
        top_segments_raw = self.extract_segments('top', output_dir, session_id)
        
        # Filter out TOO_SHORT segments - keep only GOOD and TOO_LONG
        front_valid = []
        for cow_id, segment_data in front_segments_raw.items():
            if segment_data['Flagged'] != 'TOO_SHORT':
                front_valid.append({
                    'original_id': cow_id,
                    'frames': segment_data['frames'],
                    'Flagged': segment_data['Flagged']
                })
        
        top_valid = []
        for cow_id, segment_data in top_segments_raw.items():
            if segment_data['Flagged'] != 'TOO_SHORT':
                top_valid.append({
                    'original_id': cow_id,
                    'frames': segment_data['frames'],
                    'Flagged': segment_data['Flagged']
                })
        
        print(f"\nFront segments (after filtering TOO_SHORT): {len(front_valid)}")
        print(f"Top segments (after filtering TOO_SHORT): {len(top_valid)}")
        
        # Match segments temporally
        aggregated_segments = {}
        matched_pairs = []
        
        # Check if we have matching counts
        if len(front_valid) != len(top_valid):
            print(f"\nWARNING: Front and top have different segment counts ({len(front_valid)} vs {len(top_valid)})")
            print("Matching by sequential order (fewer segments determine count)...")
        
        # Match by sequential order (pair front[i] with top[i])
        num_pairs = min(len(front_valid), len(top_valid))
        
        for i in range(num_pairs):
            front_seg = front_valid[i]
            top_seg = top_valid[i]
            
            # Check flag compatibility
            front_flag = front_seg['Flagged']
            top_flag = top_seg['Flagged']
            valid_flags = {'GOOD', 'TOO_LONG'}
            
            if front_flag not in valid_flags or top_flag not in valid_flags:
                print(f"\nSKIP pair {i}: Invalid flags - Front={front_flag}, Top={top_flag}")
                continue
            
            cow_id = f"cow_{len(matched_pairs)}"
            matched_pairs.append((cow_id, front_seg, top_seg))
            
            print(f"\nMATCH {cow_id}:")
            print(f"  Front: {front_seg['original_id']} (Flag: {front_flag})")
            print(f"  Top:   {top_seg['original_id']} (Flag: {top_flag})")
        
        # Create aggregated segments from matched pairs
        for cow_id, front_seg, top_seg in matched_pairs:
            aggregated_segments[cow_id] = {
                'front': {
                    'start_frame': front_seg['frames'][0],
                    'end_frame': front_seg['frames'][-1],
                    'duration': len(front_seg['frames']),
                    'Flagged': front_seg['Flagged']
                },
                'top': {
                    'start_frame': top_seg['frames'][0],
                    'end_frame': top_seg['frames'][-1],
                    'duration': len(top_seg['frames']),
                    'Flagged': top_seg['Flagged']
                }
            }
        
        # Extract side segments based on the matched front segments
        if aggregated_segments:
            print(f"\nExtracting segments from SIDE view...")
            # Prepare top segments data for side extraction (needs frames as a list starting with start frame)
            top_for_side = {}
            for cow_id, seg in aggregated_segments.items():
                top_start = seg['top']['start_frame']
                top_end = seg['top']['end_frame']
                top_for_side[cow_id] = {
                    'frames': list(range(top_start, top_end + 1)),
                    'Flagged': seg['top']['Flagged']
                }
            
            side_segments_raw = self.extract_segments('side', output_dir, session_id, 
                                                      top_segments=top_for_side)
            
            # Add side segments to aggregated segments
            for cow_id, segment_data in side_segments_raw.items():
                if cow_id in aggregated_segments:
                    aggregated_segments[cow_id]['side'] = {
                        'start_frame': segment_data['frames'][0],
                        'end_frame': segment_data['frames'][-1],
                        'duration': len(segment_data['frames']),
                        'Flagged': segment_data['Flagged']
                    }
        
        print(f"\n{'='*60}")
        print(f"Total matched cow pairs: {len(matched_pairs)}")
        print(f"Total cows with all views: {len(aggregated_segments)}")
        print(f"{'='*60}\n")
        
        return aggregated_segments
    
    def save_metadata(self, metadata, output_dir, session_id):
        """
        Writes the complete metadata structure to a JSON file.
        
        Args:
            metadata (dict): Metadata dictionary to save
            output_dir (str): Base output directory
            session_id (str): Session identifier
            
        Returns:
            str: Path to the saved metadata file
        """
        # Create session directory if it doesn't exist
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save to JSON file
        metadata_filename = f"metadata_{session_id}.json"
        metadata_path = os.path.join(session_dir, metadata_filename)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nMetadata saved to: {metadata_path}")
        return metadata_path    
    def _extract_side_segments(self, top_segments, side_segment_frames=30, frame_offset=40):
        """
        Helper function to extract side camera segments based on top camera segments.
        Takes side_segment_frames ending frame_offset frames before each top segment start.
        
        Top Segments extraction is more accurate and hence this was used to extract the side segments
        Args:
            top_segments (dict): Dictionary of top camera segments with format:
                                  {'cow_0': {'frames': [...], 'Flagged': '...'}, ...}
            side_segment_frames (int): Number of frames to include for side segment (default: 30).
            frame_offset (int): Number of frames before top start where side segment should end (default: 40).
        
        Returns:
            dict: Dictionary of side camera segments with format:
                  {'cow_0': {'frames': [frame_list], 'Flagged': 'GOOD'/'TOO_SHORT'}, ...}
        """
        segments = {}

        if side_segment_frames <= 0:
            raise ValueError("side_segment_frames must be greater than 0")

        if frame_offset <= 0:
            raise ValueError("frame_offset must be greater than 0")
        
        for cow_id, segment_data in top_segments.items():
            frames = segment_data['frames']
            if frames:
                top_start = frames[0]
                # Side segment ends frame_offset frames before top start and spans side_segment_frames frames.
                side_end = top_start - frame_offset
                side_start = max(1, side_end - side_segment_frames + 1)

                if side_end < 1:
                    side_frames = []
                else:
                    side_frames = list(range(side_start, side_end + 1))
                
                # Create side segment frames list
                segment_length = len(side_frames)
                
                # Check if segment meets requested minimum length.
                if segment_length >= side_segment_frames:
                    status = "GOOD"
                    print(f"Extracted SIDE cow_{cow_id[-1]}: frames {side_start} to {side_end} (total: {segment_length} frames, status: {status})")
                else:
                    status = "TOO_SHORT"
                    print(f"SIDE cow_{cow_id[-1]}: insufficient frames ({segment_length} < {side_segment_frames}, status: {status})")
                
                segments[cow_id] = {
                    'frames': side_frames,
                    'Flagged': status
                }
        
        return segments    
        
    def display_all(self, frame_delay=-1, start_frame = 0):
        frame_count = 0
        top_path, top_const = self.view_paths['top']
        side_path, side_const = self.view_paths['side']
        front_path, front_const = self.view_paths['front']
        
        front_cam = cv2.VideoCapture(front_path)
        side_cam = cv2.VideoCapture(side_path)
        top_cam = cv2.VideoCapture(top_path)
        
        #Start frame 
        front_cam.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        side_cam.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        top_cam.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        
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
        
    def test_background_sub(self, view='top'):
        background_path = f"data/processed/session_1/background_{view}.jpg"
        video_path, const = self.view_paths[view]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise ValueError(f"Could not read frame from {video_path}")
        
        
            sliced_frame = self.slice(frame, camera=const)
            subtracted_frame = self.background_subtract(background_path, sliced_frame)
            
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            #Thresholding image to make the movement more detectable, should be adjusted later
            #Threshold value
            #Noting Thresholding values for top view
            #70: good, 100: gets rid of the fence gate and lot of details of cow
            if view == 'top':
                thresh = 100
            elif view == 'front':
                thresh = 50
            cv2.threshold(subtracted_frame, thresh, 255, cv2.THRESH_BINARY, dst=subtracted_frame)
            
            active_pixels = cv2.countNonZero(cv2.cvtColor(subtracted_frame, cv2.COLOR_BGR2GRAY))
            
            #IF more than 5000 pixels are active, then assume movement
            if(view == 'top' and active_pixels > 5000) or (view == 'front' and active_pixels > 3000):
                cv2.imshow("Subtracted Frame", subtracted_frame)
                print(f"Active Pixels: {active_pixels}, Frame: {current_frame}")
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    def extract_segments(self, view, output_dir, session_id, min_frames=25, max_frames=65, top_segments=None, side_segment_frames=30):
        """
        Extracts segments from video where motion is detected, filtering for single cow segments.
        
        For 'side' view: Extracts side_segment_frames before each top segment start (default: 30).
        For 'top' and 'front' views: Uses background subtraction and motion detection.
        
        Args:
            view (str): 'top', 'side', or 'front' to specify which camera view to process.
            output_dir (str): Base output directory where segmented images will be saved.
            session_id (str): Session identifier to determine the correct session folder.
            min_frames (int): Minimum frame count for a valid segment (default: 25).
            max_frames (int): Maximum frame count for a valid segment (default: 65).
            top_segments (dict): Optional. Top camera segments to use for side view extraction.
            side_segment_frames (int): Number of frames for side view extraction (default: 30).
            
        Returns:
            dict: Dictionary of cow segments with format:
                  {'cow_0': {'frames': [frame_list], 'Flagged': 'GOOD'/'TOO_SHORT'/'TOO_LONG'}, ...}
        """
        view = view.lower()
        if view not in self.view_paths:
            raise ValueError(f"Invalid view: {view}. Must be 'top', 'side', or 'front'")
        
        # Handle side view extraction using top segments
        if view == 'side':
            if top_segments is None:
                raise ValueError("Side view extraction requires top_segments parameter")
            return self._extract_side_segments(top_segments, side_segment_frames=side_segment_frames)
        
        # Set threshold based on view
        if view == 'top':
            thresh = 75
        elif view == 'front':
            thresh = 50
        else:
            raise ValueError(f"Invalid view: {view}. Must be 'top', 'side', or 'front'")
        
        background_path = f"{output_dir}/{session_id}/background_{view}.jpg"
        video_path, video_const = self.view_paths[view]
        
        cam = cv2.VideoCapture(video_path)
        if not cam.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        cow_in_frames = []
        segments = {}
        cow_counter = 0
        motion_frame_count = 0  # Track for debugging
        
        while True:
            ret, frame = cam.read()
            
            if not ret:  # Graceful exit at end of video
                break
            
            # Get frame position BEFORE incrementing
            curr_frame = int(cam.get(cv2.CAP_PROP_POS_FRAMES))
            
            sliced_frame = self.slice(frame, camera=video_const)
            subtracted_frame = self.background_subtract(background_path, sliced_frame)
            
            cv2.threshold(subtracted_frame, thresh, 255, cv2.THRESH_BINARY, dst=subtracted_frame)
            active_pixels = cv2.countNonZero(cv2.cvtColor(subtracted_frame, cv2.COLOR_BGR2GRAY))
            
            motion_detected = (view == 'top' and active_pixels > 5000) or (view == 'front' and active_pixels > 3000)
            
            if motion_detected:
                motion_frame_count += 1
                cow_in_frames.append(curr_frame)
            else:
                # If we were tracking a cow and motion stopped, save it
                if cow_in_frames:
                    segment_length = len(cow_in_frames)
                    if min_frames <= segment_length <= max_frames:
                        status = "GOOD"
                        print(f"Saved cow_{cow_counter}: frames {cow_in_frames[0]} to {cow_in_frames[-1]} (total: {segment_length} frames, status: {status})")
                    else:
                        status = "TOO_SHORT" if segment_length < min_frames else "TOO_LONG"
                        print(f"Saved cow_{cow_counter}: frames {cow_in_frames[0]} to {cow_in_frames[-1]} (total: {segment_length} frames, status: {status})")
                    
                    segments[f"cow_{cow_counter}"] = {
                        'frames': cow_in_frames,
                        'Flagged': status
                    }
                    cow_counter += 1
                    cow_in_frames = []
        
        # Save any remaining segment after loop ends
        if cow_in_frames:
            segment_length = len(cow_in_frames)
            if min_frames <= segment_length <= max_frames:
                status = "GOOD"
                print(f"Saved cow_{cow_counter}: frames {cow_in_frames[0]} to {cow_in_frames[-1]} (total: {segment_length} frames, status: {status})")
            else:
                status = "TOO_SHORT" if segment_length < min_frames else "TOO_LONG"
                print(f"Saved cow_{cow_counter}: frames {cow_in_frames[0]} to {cow_in_frames[-1]} (total: {segment_length} frames, status: {status})")
            
            segments[f"cow_{cow_counter}"] = {
                'frames': cow_in_frames,
                'Flagged': status
            }
        
        cam.release()
        print(f"\nTotal frames with motion detected: {motion_frame_count}")
        print(f"Total segments found: {len(segments)}")
        
        return segments
                    
    def visualize_segments(self, metadata, view):
        """
        Visualizes detected segments from metadata by playing the video and highlighting frames with detected motion.
        
        Args:
            metadata (dict): Metadata dictionary with structure:
                           {
                             'segments': {
                               'cow_0': {'top': {...}, 'front': {...}, 'side': {...}},
                               ...
                             }
                           }
            view (str): 'top', 'side', or 'front' to specify which camera view to visualize.
        """
        view = view.lower()
        if view not in self.view_paths:
            raise ValueError(f"Invalid view: {view}. Must be 'top', 'side', or 'front'")
        
        video_path, video_const = self.view_paths[view]
        cam = cv2.VideoCapture(video_path)
        
        if not cam.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Create sets of frames for the specified view
        valid_frames = set()
        
        # Extract frames from metadata for the specified view
        segments = metadata.get('segments', {})
        for cow_id, views_data in segments.items():
            if view in views_data:
                segment_info = views_data[view]
                start_frame = segment_info['start_frame']
                end_frame = segment_info['end_frame']
                # Add all frames in this range
                for frame_num in range(start_frame, end_frame + 1):
                    valid_frames.add(frame_num)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"SEGMENT VISUALIZATION: {view.upper()} VIEW")
        print(f"{'='*60}")
        print("SEGMENTS:")
        for cow_id, views_data in segments.items():
            if view in views_data:
                segment_info = views_data[view]
                start_frame = segment_info['start_frame']
                end_frame = segment_info['end_frame']
                duration = segment_info['duration']
                status = segment_info.get('Flagged', 'UNKNOWN')
                print(f"  {cow_id}: Frames {start_frame:5d} - {end_frame:5d} (Duration: {duration:4d} frames, Status: {status})")
        print(f"{'='*60}\n")
        
        frame_count = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            frame_count += 1
            current_pos = int(cam.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Resize for display
            display_frame = cv2.resize(frame, (640, 480))
            
            # Check frame type and set appropriate border color
            if current_pos in valid_frames:
                # Draw green border for valid segments
                cv2.rectangle(display_frame, (5, 5), (635, 475), (0, 255, 0), 3)
                status = "VALID MOTION"
                color = (0, 255, 0)
            else:
                # Draw red border for no motion
                cv2.rectangle(display_frame, (5, 5), (635, 475), (0, 0, 255), 3)
                status = "NO MOTION"
                color = (0, 0, 255)
            
            # Add text overlay
            cv2.putText(display_frame, f"Frame: {current_pos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Status: {status}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow(f"{view.upper()} View - Segment Visualization", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Visualization stopped by user")
                break
            elif key == ord(' '):
                # Space to pause/play
                print("Paused. Press SPACE again to continue, Q to quit")
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord(' '):
                        break
                    elif key == ord('q'):
                        cam.release()
                        cv2.destroyAllWindows()
                        return
        
        cam.release()
        cv2.destroyAllWindows()
        print("Visualization complete!")
            
                    
            
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

    def run_full_pipeline(self, session_id, output_dir, background_frame_numbers=None, visualize=False, views_to_visualize=None):
        """
        Runs the complete cow gait analysis pipeline: captures backgrounds, extracts segments,
        aggregates metadata, saves to file, and optionally visualizes results.
        
        Args:
            session_id (str): Session identifier
            output_dir (str): Base output directory for processed data
            background_frame_numbers (dict): Frame numbers for background capture per view.
                                          Defaults to {'top': 0, 'front': 0} (side not needed)
            visualize (bool): Whether to visualize segments after processing
            views_to_visualize (list): List of views to visualize if visualize=True.
                                     Defaults to ['top', 'front', 'side']
        """
        if background_frame_numbers is None:
            background_frame_numbers = {'top': 0, 'front': 0}  # Side doesn't use background subtraction
        
        if views_to_visualize is None:
            views_to_visualize = ['top', 'front', 'side']
        
        print(f"\n{'='*60}")
        print(f"STARTING COW SEGMENTATION FOR ANALYSIS - Session: {session_id}")
        print(f"{'='*60}")
        
        # Create session directory
        session_dir = os.path.join(output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        print(f"Created session directory: {session_dir}")
        
        # 1. Capture background frames for views that need it (top and front only)
        print("\n1. CAPTURING BACKGROUND FRAMES...")
        for view in ['top', 'front']:  # Skip side - it uses temporal alignment instead
            frame_num = background_frame_numbers.get(view, 0)
            self.capture_background_frame(view, frame_num, output_dir, session_id)
        
        # 2. Create base metadata structure
        print("\n2. CREATING BASE METADATA STRUCTURE...")
        metadata = self.create_base_metadata(session_id)
        
        # 3. Aggregate segments from all three views
        print("\n3. EXTRACTING AND AGGREGATING SEGMENTS...")
        aggregated_segments = self.aggregate_segments(output_dir, session_id)
        
        # 4. Add aggregated segments to metadata
        metadata['segments'] = aggregated_segments
        
        # 5. Save complete metadata to file
        print("\n4. SAVING METADATA TO FILE...")
        metadata_path = self.save_metadata(metadata, output_dir, session_id)
        
        # 6. Optional visualization
        if visualize:
            print("\n5. VISUALIZING SEGMENTS...")
            # Load the saved metadata
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)
            
            # Visualize specified views
            for view in views_to_visualize:
                if view in ['top', 'side', 'front']:
                    print(f"\nVisualizing {view.upper()} view...")
                    self.visualize_segments(loaded_metadata, view)
                else:
                    print(f"Warning: Invalid view '{view}' for visualization. Skipping.")
        
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE - Session: {session_id}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"{'='*60}")
        
        return metadata_path
    
    def demo_display(self, start_frame=0, view="top"):
        path = f"data/raw/{view}/{view}.mp4"
        background_path = f"data/processed/session_1/background_top.jpg"
        
        cam = cv2.VideoCapture(path)
        
        cam.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        ret, frame = cam.read()
        sliced_frame = self.slice(frame=frame, camera=TOP)
        subtracted_frame = self.background_subtract(frame=sliced_frame, background_path=background_path)
        
        thresh = 100
        cv2.threshold(subtracted_frame, thresh, 255, cv2.THRESH_BINARY, dst=subtracted_frame)
        
        active_pixels = cv2.countNonZero(cv2.cvtColor(subtracted_frame, cv2.COLOR_BGR2GRAY))
        
        
        cv2.imshow("Frame", subtracted_frame)
        print(f"Active Pixels: {active_pixels}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        


seg = Segmenter()


# seg.display_all(frame_delay=0.5)
# seg.capture_background_frame(camera_view='front', frame_number=0, output_dir="data/processed", session_id="session_1")
# seg.create_cow_folders(output_dir="data/processed", session_id="session_1", cow_id="cow_1")
# seg.create_cow_folders(output_dir="data/processed", session_id="session_1", cow_id="cow_2")
# seg.create_cow_folders(output_dir="data/processed", session_id="session_1", cow_id="cow_3")

# cleaner.remove_cow_folders(output_dir="data/processed", session_id="session_1", cow_id="cow_1") 
# cleaner.remove_all(output_dir="data/processed", session_id="session_1", remove_session_folder=True)
    
# seg.test_background_sub("front")
# seg.display_all()

# ===== OPTIMIZED PIPELINE =====
# Run the complete cow gait analysis pipeline
# metadat_path = "data/processed/session_1/metadata_session_1.json"
# with open(metadat_path, "r") as f:
#     loaded_metadata = json.load(f)
    
# f.close()

# seg.visualize_segments(metadata=loaded_metadata, view='top')
# metadata_path = seg.run_full_pipeline(
#     session_id="session_1",
#     output_dir="data/processed",
#     background_frame_numbers={'top': 0, 'front': 0},  # Side doesn't need background
#     visualize=True,
#     views_to_visualize=['side']  # Only visualize side view as in original
# )
# seg.display_all(frame_delay=0.2)
# seg.demo_display(start_frame=1980, view="front")
# seg.capture_background_frame(camera_view="front", frame_number=0, output_dir="data/processed", session_id="session_1")