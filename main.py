from utils.video import Segmenter
from extract_frames import Extracter
import review_dashboard

extract = Extracter()
seg = Segmenter()
metadata_path = seg.run_full_pipeline(
    session_id="session_1",
    output_dir="data/processed",
    background_frame_numbers={'top': 0, 'front': 0},  # Side doesn't need background
    visualize=False,
    views_to_visualize=['top']  # Only visualize side view as in original
)