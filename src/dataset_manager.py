# dataset_manager.py

import numpy as np
import pandas as pd


class VideoDataset:
    """
    Orchestrates:
      - running feature extractor on a list of videos
      - building fixed-length sequences
      - stacking into X, y
      - providing DataFrame previews
    """

    def __init__(self, feature_extractor, sequence_builder):
        self.feature_extractor = feature_extractor
        self.sequence_builder = sequence_builder

        self.X = None
        self.y = None

    def build_from_videos(self, videos_and_labels):
        """
        videos_and_labels: list of (video_path, label)
        """
        dataset_X = []
        dataset_y = []

        for video_path, label in videos_and_labels:
            print(f"Processing Input Video: {video_path}")
            frame_features = self.feature_extractor.extract_frame_features(video_path)
            print("Raw frame_features: ", frame_features.shape)

            X_video, y_video = self.sequence_builder.build_sequences(
                frame_features=frame_features,
                label=label
            )

            dataset_X.extend(X_video)
            dataset_y.extend(y_video)
            print("-"*55)

        if dataset_X:
            self.X = np.array(dataset_X, dtype=np.float32)
            self.y = np.array(dataset_y, dtype=np.int64)
        else:
            self.X = np.empty((0, 0, 0), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.int64)

        print("Final X shape:", self.X.shape, "y shape:", self.y.shape)

    # ---------- feature naming & DataFrame helpers ----------

    def build_feature_names(self, num_joints=33):
        """
        Builds feature names based on:
          - num_joints (MediaPipe Pose = 33)
          - num_angle_features from feature_extractor.geometry
        """
        num_angle_features = self.feature_extractor.num_angle_features
        names = []
        for j in range(num_joints):
            names.append(f"j{j:02d}_x")
            names.append(f"j{j:02d}_y")
            names.append(f"j{j:02d}_conf")
            names.append(f"j{j:02d}_vx")
            names.append(f"j{j:02d}_vy")

        for a in range(num_angle_features):
            names.append(f"angle_{a:02d}")
        return names

    def sequence_to_dataframe(self, seq_idx):
        """
        Convert one sequence (all frames) to a pandas DataFrame.
        """
        if self.X is None:
            raise ValueError("Dataset is empty. Build it first.")

        X_seq = self.X[seq_idx]  # (T, D)
        y_seq = self.y[seq_idx]

        feature_names = self.build_feature_names()
        dfs = []
        for t in range(X_seq.shape[0]):
            row = X_seq[t]
            df = pd.DataFrame([row], columns=feature_names)
            df["frame"] = t
            df["label"] = y_seq
            dfs.append(df)

        df_seq = pd.concat(dfs, axis=0)
        return df_seq

    # dataset_manager.py (add inside VideoDataset class)
    def add_labeled_video(self, video_path, label):
        """
        Process a single new labeled video:
          - extract per-frame features
          - build fixed-length sequences
          - append to existing X, y
        """
        frame_features = self.feature_extractor.extract_frame_features(video_path)
        print(f"[add_labeled_video] {video_path}: raw frame_features={frame_features.shape}")

        X_video, y_video = self.sequence_builder.build_sequences(
            frame_features=frame_features,
            label=label
        )

        if not X_video:
            print(f"[add_labeled_video] No sequences generated for {video_path}.")
            return

        X_video = np.array(X_video, dtype=np.float32)
        y_video = np.array(y_video, dtype=np.int64)

        if self.X is None or self.X.size == 0:
            self.X = X_video
            self.y = y_video
        else:
            self.X = np.concatenate([self.X, X_video], axis=0)
            self.y = np.concatenate([self.y, y_video], axis=0)

        print(f"[add_labeled_video] Updated dataset shapes: X={self.X.shape}, y={self.y.shape}")

    def build_sequences_for_video(self, video_path, label=None):
        """
        Utility for single video:
          - returns sequences X_video (list of (T,D))
          - if label is not None, also returns y_video list
        Useful for testing/prediction without touching main dataset.
        """
        frame_features = self.feature_extractor.extract_frame_features(video_path)
        print(f"[build_sequences_for_video] {video_path}: raw frame_features={frame_features.shape}")

        if label is None:
            # use dummy label = 0, but ignore it in the caller
            label_tmp = 0
        else:
            label_tmp = label

        X_video, y_video = self.sequence_builder.build_sequences(
            frame_features=frame_features,
            label=label_tmp
        )
        return X_video, (y_video if label is not None else None)

    def show_samples_per_label(self, n=3):
        """
        Show first n sequences per label (first frame only) as DataFrame.
        """
        if self.X is None:
            raise ValueError("Dataset is empty. Build it first.")

        feature_names = self.build_feature_names()
        samples = []
        for seq, lbl in zip(self.X, self.y):
            row = seq[0]
            df = pd.DataFrame([row], columns=feature_names)
            df["label"] = lbl
            samples.append(df)

        df_all = pd.concat(samples, axis=0)
        print(df_all.groupby("label").head(n))

    def show_top_sequences(self, n=5):
        """
        Show first n sequences (all frames) as a DataFrame.
        """
        if self.X is None:
            raise ValueError("Dataset is empty. Build it first.")

        feature_names = self.build_feature_names()
        rows = []
        for i in range(min(n, len(self.X))):
            seq = self.X[i]
            lbl = self.y[i]
            for t, frame in enumerate(seq):
                df = pd.DataFrame([frame], columns=feature_names)
                df["frame"] = t
                df["seq"] = i
                df["label"] = lbl
                rows.append(df)
        df_preview = pd.concat(rows, axis=0)
        print(df_preview.head(20))
        return df_preview
