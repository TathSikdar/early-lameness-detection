import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy.fftpack import fft
import math

class GaitData:
    """
    Adapted from hrussel/lameness-detection for gait analysis.
    Processes keypoints to extract gait metrics.
    """
    
    def __init__(self, keypoints_csv, joints=None):
        """
        keypoints_csv: path to CSV with keypoints (video_name, frame, x1,y1,x2,y2,...)
        joints: dict of keypoint names to indices
        """
        self.keypoints_csv = keypoints_csv
        self.joints = joints or {
            'Head': ['HeadTop', 'Nose'],
            'Spine': ['Spine1', 'Spine2', 'Spine3'],
            'Legs': ['LFHoof', 'LHHoof', 'RFHoof', 'RHHoof']
        }
        self.keypoints = None
        self.n_frames = 0
        self.load_keypoints()
    
    def load_keypoints(self):
        df = pd.read_csv(self.keypoints_csv, header=None)
        # Assume columns: video_name, frame, then x,y for each kp
        n_kp = (df.shape[1] - 2) // 2
        self.keypoints = df.iloc[:, 2:].values.reshape(-1, n_kp, 2)  # (n_frames, n_kp, 2)
        self.n_frames = self.keypoints.shape[0]
    
    def get_xy_bodyparts(self, bodyparts):
        """
        Get x,y for specific bodyparts.
        bodyparts: list of kp names
        """
        # For simplicity, assume standard order: HeadTop, Nose, Spine1, etc.
        kp_indices = {
            'HeadTop': 0, 'Nose': 1, 'Spine1': 2, 'Spine2': 3, 'Spine3': 4,
            'LFHoof': 5, 'LHHoof': 6, 'RFHoof': 7, 'RHHoof': 8
        }
        indices = [kp_indices[bp] for bp in bodyparts if bp in kp_indices]
        return self.keypoints[:, indices, :]
    
    def stride_length(self, mean=True):
        """
        Calculate stride length from hoof positions.
        """
        # Simplified: distance between consecutive hoof positions
        hoofs = self.get_xy_bodyparts(['LFHoof', 'LHHoof', 'RFHoof', 'RHHoof'])
        strides = []
        for leg in range(4):
            positions = hoofs[:, leg, :]
            diffs = np.diff(positions, axis=0)
            lengths = np.linalg.norm(diffs, axis=1)
            strides.extend(lengths)
        return np.mean(strides) if mean else strides
    
    def head_bobbing(self):
        """
        Head bobbing amplitude.
        """
        head = self.get_xy_bodyparts(['HeadTop'])
        y_signal = head[:, 0, 1]  # y-coordinates
        return np.std(y_signal)  # amplitude as std dev
    
    def symmetry_score(self):
        """
        Simple symmetry: difference between left and right strides.
        """
        left_hoofs = self.get_xy_bodyparts(['LFHoof', 'LHHoof'])
        right_hoofs = self.get_xy_bodyparts(['RFHoof', 'RHHoof'])
        left_stride = self.stride_length_from_hoofs(left_hoofs)
        right_stride = self.stride_length_from_hoofs(right_hoofs)
        return abs(left_stride - right_stride)
    
    def stride_length_from_hoofs(self, hoofs):
        diffs = np.diff(hoofs[:, :, :], axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        return np.mean(lengths)
    
    def extract_features(self):
        """
        Extract all gait features.
        """
        features = {
            'stride_length': self.stride_length(),
            'head_bobbing': self.head_bobbing(),
            'symmetry': self.symmetry_score(),
            'cadence': self.n_frames / 30.0  # assuming 30 fps, rough cadence
        }
        return features