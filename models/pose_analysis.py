import numpy as np

# Configurable constants
KEYPOINT_CONF_THRESHOLD = 0.5
HEAD_BOB_MAX = 0.35  # Tune based on healthy/lame cows
SPINE_MAX = 20.0     # Tune based on healthy/lame cows
HEAD_BOB_WEIGHT = 0.5
SPINE_WEIGHT = 0.5

# Keypoint indices
HEAD = 0
NECK = 8
WITHERS = 9
TAIL_BASE = 22
LEFT_EAR = 4
RIGHT_EAR = 6


def compute_head_bob_score(frames_keypoints, frames_confs):
    angles = []
    for kps, confs in zip(frames_keypoints, frames_confs):
        if confs[LEFT_EAR] > KEYPOINT_CONF_THRESHOLD and confs[RIGHT_EAR] > KEYPOINT_CONF_THRESHOLD:
            left = kps[LEFT_EAR]
            right = kps[RIGHT_EAR]
            dx = right[0] - left[0]
            dy = right[1] - left[1]
            angle = np.arctan2(dx, dy)  # angle wrt vertical axis
            angles.append(angle)
    valid_count = len(angles)
    if valid_count == 0:
        return 0.0, 0
    angles = np.array(angles)
    angles -= np.mean(angles)
    std = np.std(angles)
    score = min(5, max(0, (std / HEAD_BOB_MAX) * 5))
    return score, valid_count


def compute_spine_score(frames_keypoints, frames_confs):
    lateral_devs = []
    vert_stds = []
    # For vertical std, collect y series for each spine kp
    y_series = {idx: [] for idx in [HEAD, NECK, WITHERS, TAIL_BASE]}
    valid_frame_count = 0
    for kps, confs in zip(frames_keypoints, frames_confs):
        # Find available/confident spine kps
        present = [idx for idx in [HEAD, NECK, WITHERS, TAIL_BASE] if confs[idx] > KEYPOINT_CONF_THRESHOLD]
        if len(present) < 2:
            continue
        valid_frame_count += 1
        # Lateral deviation
        first = present[0]
        last = present[-1]
        origin = kps[first]
        end = kps[last]
        baseline = end - origin
        norm = np.linalg.norm(baseline)
        if norm == 0:
            continue
        unit = baseline / norm
        devs = []
        for idx in present[1:-1]:
            pt = kps[idx] - origin
            cross = np.abs(np.cross(unit, pt))
            devs.append(cross)
        if devs:
            lateral_devs.append(np.mean(devs))
        # Vertical y-series
        for idx in present:
            y_series[idx].append(kps[idx][1])
    # Lateral deviation metric
    D_lateral = np.mean(np.abs(lateral_devs)) if lateral_devs else 0.0
    # Vertical std metric
    vert_vals = []
    for idx, series in y_series.items():
        if len(series) > 1:
            vert_vals.append(np.std(np.diff(series)))
    D_vertical = np.mean(vert_vals) if vert_vals else 0.0
    spine_raw = 0.5 * D_lateral + 0.5 * D_vertical
    score = min(5, max(0, (spine_raw / SPINE_MAX) * 5))
    return score, valid_frame_count


def compute_lameness_score(top_frames, model):
    """
    top_frames: list of np.ndarray (H,W,3) images
    model: loaded YOLO pose model
    Returns: dict with scores and flags
    """
    frames_keypoints = []
    frames_confs = []
    for frame in top_frames:
        results = model(frame)
        if (
            not results or len(results) == 0 or
            not hasattr(results[0], 'keypoints') or results[0].keypoints is None or
            results[0].keypoints.xy is None or len(results[0].keypoints.xy) == 0 or
            results[0].keypoints.conf is None or len(results[0].keypoints.conf) == 0
        ):
            continue
        # Defensive: check shape before accessing [0]
        if results[0].keypoints.xy.shape[0] == 0 or results[0].keypoints.conf.shape[0] == 0:
            continue
        kps = results[0].keypoints.xy[0].cpu().numpy()  # (24,2)
        confs = results[0].keypoints.conf[0].cpu().numpy()  # (24,)
        frames_keypoints.append(kps)
        frames_confs.append(confs)
    # Head bobbing
    head_bob_score, valid_head = compute_head_bob_score(frames_keypoints, frames_confs)
    # Spine alignment
    spine_score, valid_spine = compute_spine_score(frames_keypoints, frames_confs)
    # Combine
    lameness_score = round(HEAD_BOB_WEIGHT * head_bob_score + SPINE_WEIGHT * spine_score)
    lameness_score = max(0, min(5, lameness_score))
    # Flagging
    needs_review = False
    review_priority = "null"
    if valid_head < 5 or valid_spine < 5:
        needs_review = True
        review_priority = "high"
    elif lameness_score >= 5:
        needs_review = True
        review_priority = "high"
    elif lameness_score >= 3:
        needs_review = True
        review_priority = "low"
    # Output
    return {
        "head_bob_score": float(head_bob_score),
        "spine_score": float(spine_score),
        "lameness_score": int(lameness_score),
        "needs_review": needs_review,
        "review_priority": review_priority,
        "valid_frame_count_head": int(valid_head),
        "valid_frame_count_spine": int(valid_spine)
    }
