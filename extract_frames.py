import cv2
import json
import os
import shutil


class Extracter:
    """
    Extracts segmented frame sequences from videos and saves them as video clips
    organized by cow ID and camera view.
    """
    
    def __init__(self, output_dir, session_id, video_paths=None):
        """
        Initialize the Extracter.
        
        Args:
            output_dir (str): Base output directory containing session folders
            session_id (str): Session identifier (e.g., 'session_1')
            video_paths (dict): Optional dictionary of video paths with keys 'top', 'side', 'front'.
                              Defaults to 'data/raw/{view}/{view}.mp4'
        """
        self.output_dir = output_dir
        self.session_id = session_id
        
        if video_paths is None:
            self.video_paths = {
                'top': 'data/raw/top/top.mp4',
                'side': 'data/raw/side/side.mp4',
                'front': 'data/raw/front/front.mp4'
            }
        else:
            self.video_paths = video_paths
        
        self.session_dir = os.path.join(output_dir, session_id)
        self.metadata_path = os.path.join(self.session_dir, f"metadata_{session_id}.json")
    
    def _load_metadata(self):
        """
        Load metadata from the JSON file.
        
        Returns:
            dict: Metadata dictionary with structure:
                  {
                    'segments': {
                      'cow_0': {'top': {...}, 'front': {...}, 'side': {...}}, ...
                    }
                  }
        
        Raises:
            FileNotFoundError: If metadata file doesn't exist
        """
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded metadata from: {self.metadata_path}")
        return metadata
    
    def _create_cow_folder_structure(self, cow_id):
        """
        Create the folder structure for a specific cow.
        
        Args:
            cow_id (str): Cow identifier (e.g., 'cow_0')
        
        Returns:
            dict: Dictionary with paths to each view folder
        """
        cow_dir = os.path.join(self.session_dir, cow_id)
        views_dirs = {}
        
        for view in ['top', 'front', 'side', 'ear_tag']:
            view_path = os.path.join(cow_dir, view)
            os.makedirs(view_path, exist_ok=True)
            views_dirs[view] = view_path
        
        print(f"Created folder structure for {cow_id}")
        return views_dirs
    
    def _extract_and_save_clip(self, video_path, start_frame, end_frame, output_path, view):
        """
        Extract frames from a video and save as a video clip.
        
        Args:
            video_path (str): Path to the source video file
            start_frame (int): Starting frame number
            end_frame (int): Ending frame number (inclusive)
            output_path (str): Path where the output video will be saved
            view (str): Camera view name ('top', 'front', 'side') - for logging
        
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            # Open the video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"ERROR: Could not open video: {video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            if not out.isOpened():
                print(f"ERROR: Could not create VideoWriter for: {output_path}")
                cap.release()
                return False
            
            # Extract and write frames
            frame_count = 0
            for frame_num in range(start_frame, end_frame + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"WARNING: Could not read frame {frame_num} from {view} view")
                    break
                
                out.write(frame)
                frame_count += 1
            
            # Release resources
            cap.release()
            out.release()
            
            duration_frames = end_frame - start_frame + 1
            duration_seconds = duration_frames / fps
            print(f"  Extracted {view}: {frame_count} frames ({duration_seconds:.2f}s) -> {output_path}")
            
            return True
        
        except Exception as e:
            print(f"ERROR extracting clip from {video_path}: {str(e)}")
            return False
    
    def extract_all_segments(self):
        """
        Extract all segments from the metadata and save them as video clips.
        
        Returns:
            dict: Summary statistics of extraction (cows processed, clips created, etc.)
        """
        print(f"\n{'='*60}")
        print(f"STARTING FRAME EXTRACTION - Session: {self.session_id}")
        print(f"{'='*60}")
        
        # Load metadata
        metadata = self._load_metadata()
        segments = metadata.get('segments', {})
        
        if not segments:
            print("No segments found in metadata!")
            return {'success': False, 'reason': 'No segments in metadata'}
        
        stats = {
            'total_cows': len(segments),
            'cows_processed': 0,
            'clips_created': 0,
            'clips_failed': 0
        }
        
        # Process each cow
        for cow_id, views_data in segments.items():
            print(f"\nProcessing {cow_id}...")
            
            # Create folder structure for the cow
            views_dirs = self._create_cow_folder_structure(cow_id)
            
            # Extract segments for each view
            for view in ['top', 'front', 'side']:
                if view not in views_data:
                    print(f"  WARNING: No {view} segment data for {cow_id}")
                    continue
                
                segment_info = views_data[view]
                start_frame = segment_info['start_frame']
                end_frame = segment_info['end_frame']
                status = segment_info.get('Flagged', 'UNKNOWN')
                
                print(f"  {view.upper()}: Frames {start_frame}-{end_frame} (Status: {status})")
                
                # Extract and save clip
                video_path = self.video_paths[view]
                output_filename = f"{cow_id}_{view}_segment.mp4"
                output_path = os.path.join(views_dirs[view], output_filename)
                
                success = self._extract_and_save_clip(video_path, start_frame, end_frame, output_path, view)
                
                if success:
                    stats['clips_created'] += 1
                else:
                    stats['clips_failed'] += 1
            
            stats['cows_processed'] += 1
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE - Session: {self.session_id}")
        print(f"{'='*60}")
        print(f"Cows processed: {stats['cows_processed']}/{stats['total_cows']}")
        print(f"Clips created: {stats['clips_created']}")
        print(f"Clips failed: {stats['clips_failed']}")
        print(f"{'='*60}\n")
        
        stats['success'] = True
        return stats


class Cleaner:
    """
    Removes extracted cow folders while preserving background images and metadata.
    """
    
    def __init__(self, output_dir, session_id):
        """
        Initialize the Cleaner.
        
        Args:
            output_dir (str): Base output directory containing session folders
            session_id (str): Session identifier (e.g., 'session_1')
        """
        self.output_dir = output_dir
        self.session_id = session_id
        self.session_dir = os.path.join(output_dir, session_id)
    
    def remove_cow_folder(self, cow_id):
        """
        Remove a specific cow's folder and all its contents.
        
        Args:
            cow_id (str): Cow identifier (e.g., 'cow_0')
        
        Returns:
            bool: True if successfully removed, False otherwise
        """
        cow_dir = os.path.join(self.session_dir, cow_id)
        
        if not os.path.exists(cow_dir):
            print(f"Cow folder does not exist: {cow_dir}")
            return False
        
        try:
            shutil.rmtree(cow_dir)
            print(f"Removed cow folder: {cow_dir}")
            return True
        except Exception as e:
            print(f"ERROR removing cow folder: {str(e)}")
            return False
    
    def remove_all_cows(self):
        """
        Remove all cow folders in the session.
        
        This preserves:
        - background_top.jpg
        - background_front.jpg
        - metadata_session_x.json
        
        Returns:
            dict: Summary of removal (cows removed, failed attempts, etc.)
        """
        print(f"\n{'='*60}")
        print(f"CLEANING UP EXTRACTED FRAMES - Session: {self.session_id}")
        print(f"{'='*60}")
        
        if not os.path.exists(self.session_dir):
            print(f"Session directory does not exist: {self.session_dir}")
            return {'success': False, 'reason': 'Session directory not found'}
        
        stats = {
            'cows_removed': 0,
            'cows_failed': 0
        }
        
        # List all items in session directory
        items = os.listdir(self.session_dir)
        
        # Remove cow folders (cow_X, cow_0, etc.)
        for item in items:
            item_path = os.path.join(self.session_dir, item)
            
            # Only remove directories that look like cow folders
            if os.path.isdir(item_path) and item.startswith('cow_'):
                if self.remove_cow_folder(item):
                    stats['cows_removed'] += 1
                else:
                    stats['cows_failed'] += 1
        
        # Verify that background images and metadata are still present
        preserved_files = []
        for filename in ['background_top.jpg', 'background_front.jpg']:
            filepath = os.path.join(self.session_dir, filename)
            if os.path.exists(filepath):
                preserved_files.append(filename)
        
        metadata_path = os.path.join(self.session_dir, f"metadata_{self.session_id}.json")
        if os.path.exists(metadata_path):
            preserved_files.append(f"metadata_{self.session_id}.json")
        
        print(f"\n{'='*60}")
        print(f"CLEANUP COMPLETE - Session: {self.session_id}")
        print(f"{'='*60}")
        print(f"Cow folders removed: {stats['cows_removed']}")
        print(f"Removal failures: {stats['cows_failed']}")
        print(f"Preserved files: {', '.join(preserved_files)}")
        print(f"{'='*60}\n")
        
        stats['success'] = True
        return stats


# Example usage
if __name__ == "__main__":
    # Extract frames from metadata
    extracter = Extracter(
        output_dir="data/processed",
        session_id="session_1"
    )
    extraction_stats = extracter.extract_all_segments()
    
    # Example: Clean up extracted frames (if needed)
    # cleaner = Cleaner(
    #     output_dir="data/processed",
    #     session_id="session_1"
    # )
    # cleanup_stats = cleaner.remove_all_cows()
