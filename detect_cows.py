import os
import json
# Utility: Create or update cow JSON in cow folder
def create_or_update_cow_json(
    cow_id: str,
    session_id: str,
    video_path: str,
    cow_folder: str,
    **kwargs
) -> str:
    json_path = os.path.join(cow_folder, "lameness_analysis.json")
    # If file exists, load and update; else create new
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        cow_obj = Cow.from_json(json.dumps(data))
        # Update fields from kwargs
        for k, v in kwargs.items():
            setattr(cow_obj, k, v)
    else:
        cow_obj = Cow(
            cow_id=cow_id,
            session_id=session_id,
            video_path=video_path,
            **kwargs
        )
    # Save to file
    with open(json_path, "w") as f:
        f.write(cow_obj.to_json())
    return json_path

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Any
import datetime as dt

@dataclass
class Cow:
    # Identification and session
    cow_id: str
    session_id: str
    video_path: str
    timestamp: str = field(default_factory=lambda: dt.datetime.now().isoformat())

    # Ear tag prediction/correction
    predicted_ear_tag_id: Optional[str] = None
    predicted_ear_tag_confidence: Optional[float] = None
    corrected_ear_tag_id: Optional[str] = None

    # Lameness scoring
    lameness_score: Optional[int] = None
    correction_value: Optional[int] = 0
    corrected_lameness_score: Optional[int] = None
    correction_timestamp: Optional[str] = None

    # Metrics
    head_bob_score: Optional[float] = None
    spine_score: Optional[float] = None
    valid_frame_count_head: Optional[int] = None
    valid_frame_count_spine: Optional[int] = None

    # Review flags
    needs_review: bool = False
    review_priority: str = "null"
    notes: Optional[str] = ""

    # Model/version info
    model_version: str = "V0.0"

    # Optional: frame-level results for debugging/analytics
    frame_level_results: Optional[List[Any]] = field(default_factory=list)

    def to_json(self):
        import json
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(json_str):
        import json
        data = json.loads(json_str)
        return Cow(**data)


print(f"dt.time.max(): {dt.time}")


print("-----------------------------")
print(type(10))