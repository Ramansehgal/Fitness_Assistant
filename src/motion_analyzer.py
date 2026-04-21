import numpy as np
import mediapipe as mp
from collections import defaultdict

from feature_extractor import PoseFeatureExtractor

class MotionAnalyzer:
    def __init__(self, feature_extractor, angle_triplets, dominant_bone_map, sample_fps=10):
        """
        valid_angle_ranges: dict
            e.g. {"knee": (70,110), "hip": (60,120)}
        fps: sampling fps
        """
        self.RAD2DEG = 180.0 / np.pi
        self.mp = mp.solutions.pose

        # self.valid_angle_ranges = valid_angle_ranges
        self.feature_extractor = feature_extractor
        self.angle_triplets = angle_triplets        # {angle_id: (a,b,c)}
        self.dominant_bone_map = dominant_bone_map  # {angle_id: (u,v)}
        self.sample_fps = sample_fps

    def build_frames_timestamp_per_video(self, video_path):
        """
        Returns a list of frames, one per integer second:
        [
          {
            "timestamp": int,
            "angles": {angle_id: float},
            "bone_lengths": {angle_id: float}
          }
        ]
        """
        raw = self.feature_extractor.extract_frame_features(video_path)
        # raw shape: (T, D)

        frames = []
        frames_per_sec = self.sample_fps
        T = raw.shape[0]
        total_secs = T // frames_per_sec

        for sec in range(total_secs):
            start = sec * frames_per_sec
            end = start + frames_per_sec
            chunk = raw[start:end]

            angles_sec = {}
            bones_sec = {}

            for angle_id, (a, b, c) in self.angle_triplets.items():
                # angle index already known in feature vector
                angles_sec[angle_id] = float(
                    np.mean(chunk[:, self.feature_extractor.angle_index(angle_id)]) * self.RAD2DEG 
                )

                u, v = self.dominant_bone_map[angle_id]
                bones_sec[angle_id] = float(
                    np.mean(chunk[:, self.feature_extractor.bone_index(u, v)]) 
                )

            frames.append({
                "timestamp": sec,
                "angles": angles_sec,
                "bone_lengths": bones_sec
            })

        return frames
    
    def analyze_segment(self, frames, valid_angle_ranges):
        """
        frames: list of dicts:
          {
            "timestamp": int,
            "angles": {angle_id: value},
            "bone_lengths": {angle_id: value}
          }
        """

        joint_angles = defaultdict(list)
        joint_lengths = defaultdict(list)
        timestamps = []

        # 1. Collect data
        for f in frames:
            timestamps.append(f["timestamp"])
            for j, a in f["angles"].items():
                joint_angles[j].append(a)
            for j, l in f["bone_lengths"].items():
                joint_lengths[j].append(l)

        # 2. Deviation score (angle + bone)
        deviation_scores = {}
        for j in joint_angles:
            angle_std = np.std(joint_angles[j])
            bone_std = np.std(joint_lengths[j])
            deviation_scores[j] = angle_std + bone_std

        # 3. Major impacted joints (Top IQR)
        scores = np.array(list(deviation_scores.values()))
        threshold = np.percentile(scores, 75)

        major_joints = [
            j for j, s in deviation_scores.items()
            if s >= threshold
        ]
        major_joints_name = [self.mp.PoseLandmark(self.angle_triplets[j][1]).name for j in major_joints]
        
        # 4. Group angle IDs by joint name (CENTER of triplet)
        joint_to_angle_ids = defaultdict(list)
        for aid in major_joints:
            _, center_joint, _ = self.angle_triplets[aid]
            joint_name = self.mp.PoseLandmark(center_joint).name
            joint_to_angle_ids[joint_name].append(aid)

        # 5. Detailed statistics
        joint_stats = {}
        for j in major_joints:
            angles = np.array(joint_angles[j])
            lengths = np.array(joint_lengths[j])
            t = np.array(timestamps)

            min_idx = np.argmin(angles)
            max_idx = np.argmax(angles)

            low, high = valid_angle_ranges.get(j, (-np.inf, np.inf))
            in_range = (angles >= low) & (angles <= high)

            joint_stats[j] = {
                # per-timestamp series
                "angle_per_second": angles.tolist(),
                "bone_length_per_second": lengths.tolist(),

                # angle stats
                "mean_angle": float(np.mean(angles)),
                "min_angle": float(angles[min_idx]),
                "max_angle": float(angles[max_idx]),
                "angle_std": float(np.std(angles)),

                # timing
                "t_min": int(t[min_idx]),
                "t_max": int(t[max_idx]),

                # range consistency
                "out_of_range_pct": float(100 * (1 - np.mean(in_range))),
                "in_range_pct": float(100 * np.mean(in_range)),

                # bone stats
                "mean_bone_length": float(np.mean(lengths)),
                "bone_length_delta": float(np.max(lengths) - np.min(lengths)),
                "bone_length_std": float(np.std(lengths)),
            }

        # 5. Rep quality
        good_frames = 0
        for i in range(len(frames)):
            ok = True
            for j in major_joints:
                a = joint_angles[j][i]
                low, high = valid_angle_ranges.get(j, (-np.inf, np.inf))
                if not (low <= a <= high):
                    ok = False
                    break
            if ok:
                good_frames += 1

        good_ratio = good_frames / len(frames)

        return {
            "major_impacted_joints": major_joints,
            "major_impacted_joint_names": major_joints_name,
            "major_impacted_joints_grouped": dict(joint_to_angle_ids),
            "joint_statistics": joint_stats,
            "rep_quality": "good" if good_ratio >= 0.8 else "bad",
            "rep_score": round(good_ratio, 3)
        }

    def format_segment_for_llm(self, segment_output):
        """
        Converts analyze_segment output into LLM-ready structure
        """

        formatted = {
            "summary": {
                "rep_quality": segment_output["rep_quality"],
                "rep_score": segment_output["rep_score"]
            },
            "major_joints": {}
        }

        grouped = segment_output["major_impacted_joints_grouped"]
        stats = segment_output["joint_statistics"]

        for joint_name, angle_ids in grouped.items():
            joint_entry = {
                "angles": [],
                "bone_lengths": [],
                "timestamps": set(),
            }

            for aid in angle_ids:
                s = stats[aid]

                joint_entry["angles"].append({
                    "angle_id": aid,
                    "mean_deg": s["mean_angle"],
                    "min_deg": s["min_angle"],
                    "max_deg": s["max_angle"],
                    "std_deg": s["angle_std"],
                    "t_min": s["t_min"],
                    "t_max": s["t_max"],
                    "out_of_range_pct": s["out_of_range_pct"],
                })

                joint_entry["bone_lengths"].append({
                    "mean": s["mean_bone_length"],
                    "delta": s["bone_length_delta"],
                    "std": s["bone_length_std"],
                })

                joint_entry["timestamps"].update([s["t_min"], s["t_max"]])

            joint_entry["timestamps"] = sorted(joint_entry["timestamps"])
            formatted["major_joints"][joint_name] = joint_entry

        return formatted

    def build_llm_timeline_input(self,
        segment_output,
        segment_time_range,
        valid_angle_ranges,
        max_joints_per_timestep=2
    ):
        """
        Builds per-timestamp summaries for LLM reasoning.
        """

        major_joints = segment_output["major_joints"]
        # joint_stats = segment_output["joint_statistics"]

        T_start, T_end = segment_time_range
        timestamps = list(range(T_start, T_end + 1))

        timeline = []

        for idx, t in enumerate(timestamps):
            joint_entries = []

            for joint_name, data in major_joints.items():
                if idx >= len(data["timestamps"]):
                    continue

                # Angle stats (take dominant angle per joint)
                angle = data["angles"][0]
                angle_val = angle["mean_deg"]

                low, high = valid_angle_ranges.get(joint_name, (-float("inf"), float("inf")))
                in_range = low <= angle_val <= high

                bone = data["bone_lengths"][0]

                joint_entries.append({
                    "joint": joint_name,
                    "mean_angle": round(angle_val, 1),
                    "mean_bone_length": round(bone["mean"], 2),
                    "in_range": in_range
                })

            # Sort by deviation (optional but recommended)
            joint_entries = joint_entries[:max_joints_per_timestep]

            timeline.append({
                "time": t,
                "joints": joint_entries
            })

        return timeline

'''
class MotionAnalyzer:
    def __init__(self, exercise_profile: dict):
        self.profile = exercise_profile

    def analyze_segment(self, frames):
        """
        frames: list of dicts with keys:
            - timestamp
            - angles
            - bone_lengths
        """
        joint_angles = defaultdict(list)
        joint_lengths = defaultdict(list)
        timestamps = []

        for f in frames:
            timestamps.append(f["timestamp"])
            for j, a in f["angles"].items():
                joint_angles[j].append(a)
            for j, l in f["bone_lengths"].items():
                joint_lengths[j].append(l)

        # 1. Compute deviation score per joint
        deviation_scores = {}
        for j in joint_angles:
            angles = np.array(joint_angles[j])
            lengths = np.array(joint_lengths.get(j, []))

            angle_dev = np.std(angles)
            bone_dev = np.std(lengths) if len(lengths) else 0.0
            deviation_scores[j] = angle_dev + bone_dev

        # 2. Select major impacted joints (top IQR)
        scores = np.array(list(deviation_scores.values()))
        threshold = np.percentile(scores, 75)

        major_joints = [
            j for j, s in deviation_scores.items()
            if s >= threshold
        ]

        # 3. Compute detailed stats
        joint_stats = {}
        for j in major_joints:
            angles = np.array(joint_angles[j])
            lengths = np.array(joint_lengths[j])
            t = np.array(timestamps)

            min_idx = np.argmin(angles)
            max_idx = np.argmax(angles)

            low, high = self.valid_angle_ranges.get(j, (-np.inf, np.inf))
            out_pct = np.mean((angles < low) | (angles > high)) * 100

            joint_stats[j] = {
                "mean_angle": float(np.mean(angles)),
                "min_angle": float(angles[min_idx]),
                "max_angle": float(angles[max_idx]),
                "t_min": float(t[min_idx]),
                "t_max": float(t[max_idx]),
                "mean_bone_length": float(np.mean(lengths)),
                "bone_length_delta": float(np.max(lengths) - np.min(lengths)),
                "out_of_range_pct": float(out_pct)
            }

        # 4. Rep quality
        good_frames = 0
        for i in range(len(frames)):
            ok = True
            for j in major_joints:
                a = joint_angles[j][i]
                low, high = self.valid_angle_ranges.get(j, (-np.inf, np.inf))
                if not (low <= a <= high):
                    ok = False
                    break
            if ok:
                good_frames += 1

        good_ratio = good_frames / len(frames)

        rep_quality = "good" if good_ratio >= 0.8 else "bad"

        return {
            "major_impacted_joints": major_joints,
            "joint_statistics": joint_stats,
            "rep_quality": rep_quality,
            "rep_score": round(good_ratio, 3)
        }

    def analyze_segment(self, frame_metrics: list):
        """
        frame_metrics: list of dicts
        Each dict contains joint angles/distances for one frame
        """

        results = {}

        for angle_name in self.profile["angles"]:
            values = [f[angle_name] for f in frame_metrics if angle_name in f]

            results[angle_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "outlier_pct": float(
                    np.mean(np.abs(values - np.mean(values)) > 2*np.std(values))
                )
            }

        return results
'''
