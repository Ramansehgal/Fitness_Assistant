# feature_extractor.py

import cv2
import numpy as np
import mediapipe as mp
from pose_geometry import PoseGeometry


class PoseFeatureExtractor:
    """
    Handles:
      - Reading videos
      - Running MediaPipe Pose
      - Normalizing coords
      - Building per-frame feature vectors: [x,y,conf,vx,vy,...angles]
    """

    def __init__(self, sample_fps=10, max_frames=None, exclude_angle_centers=None):
        self.sample_fps = sample_fps
        self.max_frames = max_frames

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.geometry = PoseGeometry(self.mp_pose, exclude_angle_centers=exclude_angle_centers)

        # Angle triplets computed by PoseGeometry
        self.angle_triplets = self.geometry.angle_triplets
        self.num_angle_features = self.geometry.num_angle_features

    def _build_frame_features(self, coords_norm, confs, prev_coords_norm=None):
        """
        Build per-frame feature vector from normalized coords and confidences.
        Features per joint: [x_norm, y_norm, conf, vx, vy]
        Then angle features from angle triplets.

        coords_norm: (V,2)
        confs: (V,)
        prev_coords_norm: (V,2) or None
        Returns: 1D feature vector
        """
        V = coords_norm.shape[0]

        # velocities
        if prev_coords_norm is None:
            v = np.zeros_like(coords_norm)
        else:
            v = coords_norm - prev_coords_norm

        joint_feats = []
        for j in range(V):
            x, y = coords_norm[j]
            conf = confs[j]
            vx, vy = v[j]
            joint_feats.extend([x, y, conf, vx, vy])

        angle_feats = []
        for (a, b, c) in self.angle_triplets:
            ang = self.geometry.compute_angle(coords_norm[a], coords_norm[b], coords_norm[c])
            angle_feats.append(ang)

        return np.array(joint_feats + angle_feats, dtype=np.float32)

    def extract_frame_features(self, video_path):
        """
        Extract per-frame features from a video.
        Returns:
            frame_features: np.array of shape (T, D)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open:", video_path)
            return np.empty((0, 0), dtype=np.float32)

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30.0

        step = max(1, int(round(orig_fps / self.sample_fps)))

        frame_features = []
        frame_idx = 0
        prev_coords_norm = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % step != 0:
                continue

            if self.max_frames is not None and len(frame_features) >= self.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if not results.pose_landmarks:
                prev_coords_norm = None
                continue

            landmarks = results.pose_landmarks.landmark
            V = len(landmarks)
            coords = np.zeros((V, 2), dtype=np.float32)
            confs = np.zeros((V,), dtype=np.float32)

            for i, lm in enumerate(landmarks):
                coords[i, 0] = lm.x
                coords[i, 1] = lm.y
                confs[i] = lm.visibility

            coords_norm = self.geometry.normalize_coords(coords)
            feat = self._build_frame_features(coords_norm, confs, prev_coords_norm)
            frame_features.append(feat)

            prev_coords_norm = coords_norm

        cap.release()
        cv2.destroyAllWindows()

        if not frame_features:
            return np.empty((0, 0), dtype=np.float32)

        frame_features = np.stack(frame_features, axis=0)  # (T, D)
        return frame_features
