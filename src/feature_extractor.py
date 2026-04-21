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
        
        self.num_joints = 33
        self.joint_feature_dim = 5  # x,y,conf,vx,vy
        self.RAD2DEG = 180.0 / np.pi

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.geometry = PoseGeometry(self.mp_pose, exclude_angle_centers=exclude_angle_centers)

        self.joint_feat_start = 0
        self.angle_feat_start = self.num_joints * self.joint_feature_dim
        self.bone_feat_start = self.angle_feat_start + self.geometry.num_angle_features

        self.sample_fps = sample_fps
        self.max_frames = max_frames

        # Angle triplets computed by PoseGeometry
        self.angle_triplets = self.geometry.angle_triplets
        self.num_angle_features = self.geometry.num_angle_features

        # map: angle_id → (a,b,c)
        self.angle_id_map = {i: triplet for i, triplet in enumerate(self.angle_triplets)}

        # dominant bone = (center_joint, distal_joint)
        self.dominant_bone_map = {
            i: (triplet[1], triplet[2])   # (b, c)
            for i, triplet in self.angle_id_map.items()
        }

    def angle_index(self, angle_id):
        """
        Returns column index for given angle_id
        """
        return self.angle_feat_start + angle_id


    def bone_index(self, u, v):
        """
        Returns column index for bone (u,v)
        """
        # find which angle owns this bone
        for angle_id, pair in self.dominant_bone_map.items():
            if pair == (u, v):
                return self.bone_feat_start + angle_id

        raise KeyError(f"No dominant bone mapping for ({u},{v})")
    
    def get_series(self, frame_features, idx, is_bone_id=False):
        """
        Returns (T,) series for angle_id or bone_id
        """
        offset = self.bone_feat_start if is_bone_id else self.angle_feat_start
        return frame_features[:, offset + idx] * self.RAD2DEG

    # --------------------------------------------------
    # Generic stats (angle or bone)
    # --------------------------------------------------
    def series_stats(self, frame_features, idx, timestamps=None, is_bone_id=False):
        series = self.get_series(frame_features, idx, is_bone_id)

        min_idx = np.argmin(series)
        max_idx = np.argmax(series)

        stats = {
            "idx" : idx,
            "series" : series,
            "mean": float(np.mean(series)),
            "min": float(series[min_idx]),
            "max": float(series[max_idx]),
            "std": float(np.std(series)),
        }

        if timestamps is not None:
            stats["t_min"] = float(timestamps[min_idx])
            stats["t_max"] = float(timestamps[max_idx])

        return stats

    # --------------------------------------------------
    # All angle stats
    # --------------------------------------------------
    def all_angle_stats(self, frame_features, timestamps=None):
        stats = {}
        for a in range(self.num_angle_features):
            stats[f"angle_{a:02d}"] = self.series_stats(
                frame_features,
                a,
                timestamps=timestamps,
                is_bone_id=False
            )
        return stats

    # --------------------------------------------------
    # All bone stats
    # --------------------------------------------------
    def all_bone_stats(self, frame_features, timestamps=None):
        stats = {}
        for b in range(self.num_angle_features):
            stats[f"bone_{b:02d}"] = self.series_stats(
                frame_features,
                b,
                timestamps=timestamps,
                is_bone_id=True
            )
        return stats
    
    # ---------------------------
    # Optional DataFrame view
    # ---------------------------
    def to_dataframe(self, frame_features, timestamps=None):
        """
        Convert to DataFrame ONLY when needed
        """
        cols = []

        # joint features
        for j in range(self.num_joints):
            for f in ["x", "y", "conf", "vx", "vy"]:
                cols.append(f"j{j:02d}_{f}")

        # angle features
        for a in range(self.num_angle_features):
            cols.append(f"angle_{a:02d}")

        df = pd.DataFrame(frame_features, columns=cols)

        if timestamps is not None:
            df.insert(0, "time", timestamps)

        return df
    
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

        angle_feats = []
        bone_feats = []

        for angle_id, (a, b, c) in self.angle_id_map.items():
            ang = self.geometry.compute_angle(coords_norm[a], coords_norm[b], coords_norm[c])
            angle_feats.append(ang)

            u, v = self.dominant_bone_map[angle_id]
            bone_len = np.linalg.norm(coords_norm[u] - coords_norm[v])
            bone_feats.append(bone_len)

        return np.array(joint_feats + angle_feats + bone_feats, dtype=np.float32)

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

        print("Video FPS reported by OpenCV:", orig_fps)
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
