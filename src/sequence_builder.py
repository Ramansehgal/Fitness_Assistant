# sequence_builder.py

import numpy as np
import random


class FixedLengthSequenceBuilder:
    """
    Takes per-frame features (T, D) and:
      - groups frames per second using fps
      - decides how many frames to take from each second
      - builds sequences of fixed length `target_len` (e.g. 100)
      - can generate multiple sequences per video (augmentation)
    """

    def __init__(self, fps=10, target_len=100, num_sequences=5, debug=False):
        self.fps = fps
        self.target_len = target_len
        self.num_sequences = num_sequences
        self.debug = debug

    def build_sequences(self, frame_features, label):
        """
        frame_features: (T, D)
        label: int

        Returns:
            X_seqs: list of np.array of shape (target_len, D)
            y_seqs: list of labels
        """
        frame_features = np.asarray(frame_features, dtype=np.float32)
        if frame_features.ndim != 2:
            raise ValueError(f"frame_features must be 2D (T,D), got {frame_features.shape}")
        T, D = frame_features.shape

        if T == 0:
            if self.debug:
                print("[FixedLengthSequenceBuilder] Empty frame_features, skipping.")
            return [], []

        duration_s = T // self.fps
        if duration_s == 0:
            if self.debug:
                print("[FixedLengthSequenceBuilder] Not enough frames for even 1 second, skipping.")
            return [], []

        usable_T = duration_s * self.fps
        frame_features = frame_features[:usable_T]
        T = usable_T

        if self.debug:
            print(f"[FixedLengthSequenceBuilder] T={T}, D={D}, fps={self.fps}, duration_s={duration_s}")

        # if video longer than target_len seconds (rare), clamp duration
        if duration_s > self.target_len:
            if self.debug:
                print(f"[WARN] duration_s={duration_s} > target_len={self.target_len}, "
                      f"clamping to first {self.target_len} seconds.")
            duration_s = self.target_len
            usable_T = duration_s * self.fps
            frame_features = frame_features[:usable_T]
            T = usable_T

        # 1) group frame indices per second
        sec_to_frames = {sec: [] for sec in range(duration_s)}
        for idx in range(T):
            sec = idx // self.fps
            if sec < duration_s:
                sec_to_frames[sec].append(idx)

        if self.debug:
            for sec in range(duration_s):
                print(f"  second {sec}: {len(sec_to_frames[sec])} frames -> idx {sec_to_frames[sec]}")

        # 2) decide frames per sec to reach target_len
        if self.target_len < duration_s:
            raise ValueError("target_len must be >= duration_s")

        base = self.target_len // duration_s
        extra = self.target_len % duration_s

        per_sec_counts = []
        for sec in range(duration_s):
            count = base + (1 if sec < extra else 0)
            per_sec_counts.append(count)

        if self.debug:
            print(f"  base={base}, extra={extra}, per_sec_counts={per_sec_counts}, sum={sum(per_sec_counts)}")

        X_seqs = []
        y_seqs = []

        # 3) build sequences
        for s in range(self.num_sequences):
            chosen_indices = []
            for sec, count in enumerate(per_sec_counts):
                frame_ids = sec_to_frames[sec]
                L = len(frame_ids)

                if L == 0:
                    continue

                if L >= count:
                    chosen = random.sample(frame_ids, count)
                    chosen.sort()
                else:
                    pos = np.linspace(0, L - 1, count)
                    idx_local = np.round(pos).astype(int)
                    chosen = [frame_ids[i] for i in idx_local]

                chosen_indices.extend(chosen)

            chosen_indices = sorted(chosen_indices)
            seq = frame_features[chosen_indices]  # (target_len, D)

            if self.debug and s == 0:
                print(f"  example sequence idx (seq 0): {chosen_indices[:30]} ... total={len(chosen_indices)}")
                print(f"  example sequence shape: {seq.shape}")

            X_seqs.append(seq)
            y_seqs.append(label)

        if self.debug:
            print(f"[FixedLengthSequenceBuilder] generated {len(X_seqs)} sequences for this video.\n")

        return X_seqs, y_seqs
