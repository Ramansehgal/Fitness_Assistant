import os
import cv2
import math
import random
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import defaultdict
from itertools import combinations

# ---- video setup ----
video_path = "data/videos/pushup2.mp4"
cap = cv2.VideoCapture(video_path)

print("CWD:", os.getcwd())
print("Exists (video_path):", os.path.exists(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
print("Video FPS reported by OpenCV:", fps)

if not cap.isOpened():
    print("❌ Could not open video:", video_path)
    raise SystemExit

def compute_angle(a, b, c):
    """
    Angle at point b formed by vectors ba and bc, returns radians.
    a, b, c are np.array([x,y]).
    """
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba) + 1e-8
    nb = np.linalg.norm(bc) + 1e-8
    ba /= na
    bc /= nb
    cosang = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return math.acos(cosang)

def normalize_coords(coords, pose_connections, eps=1e-6):
    """
    coords: (V,2) array of MediaPipe normalized coords (0..1, 0..1)
    pose_connections: list of (a,b) pairs defining skeleton graph
                      e.g. mp_pose.POSE_CONNECTIONS
    
    Returns:
        coords_norm: (V,2) root-relative, scale-normalized coords

    Steps:
      1. Find a "root" joint: mid-hip if possible, else mean of all joints
      2. Compute coords relative to root 
      3. Compute scale: average bone length of skeleton edges
      4. Divide coords by scale
    """

    V = coords.shape[0]

    # 1) Get root joint: mid-hip if exists ----
    # MediaPipe BlazePose hip indices (if available)
    L_HIP = 23
    R_HIP = 24

    if L_HIP < V and R_HIP < V:
        root = (coords[L_HIP] + coords[R_HIP]) / 2.0
    else:
        # fallback: center of mass
        root = coords.mean(axis=0)

    # subtract root -> root-relative coordinates
    coords_rel = coords - root

    # 2) compute scale factor based on bone lengths ----
    # collect lengths for all edges
    lengths = []
    for (a, b) in pose_connections:
        if a < V and b < V:
            d = np.linalg.norm(coords[a] - coords[b])
            if d > eps:
                lengths.append(d)

    if len(lengths) == 0:
        scale = 1.0                     # fallback: unit scale
    else:
        scale = np.mean(lengths)

    if scale < eps:
        scale = 1.0                     # avoid division by zero

    coords_norm = coords_rel / scale
    return coords_norm

def resample_to_length(feats, target_len):
    """
    feats: (T, D) per-frame features
    target_len: desired number of frames

    Returns: (target_len, D)
    - If T > target_len: uniformly downsample.
    - If T < target_len: uniformly sample with repetition
      (some frames repeated, but still real frames).
    """
    T, D = feats.shape
    if T == 0:
        return np.zeros((target_len, D), dtype=np.float32)

    if T == target_len:
        return feats.copy()

    # choose indices via linspace from 0..T-1
    idxs = np.linspace(0, T - 1, target_len)
    idxs = np.round(idxs).astype(int)
    idxs = np.clip(idxs, 0, T - 1)

    return feats[idxs]

def build_angle_triplets(mp_pose, exclude_centers=None):
    
    if exclude_centers is None:
        exclude_centers = []

    # Undirected adjacency from POSE_CONNECTIONS
    adj = defaultdict(set)
    for a, c in mp_pose.POSE_CONNECTIONS:
        adj[a].add(c)
        adj[c].add(a)

    triplets = []
    for b, neighbors in adj.items():
        if b in exclude_centers:
            continue
        neighbors = list(neighbors)
        if len(neighbors) < 2:
            continue
        n = len(neighbors)
        for i in range(n):
            for j in range(i+1, n):
                a = neighbors[i]
                c = neighbors[j]
                triplets.append((a, b, c))
    return triplets

def build_frame_features(coords_norm, confs, prev_coords_norm=None):
    """
    Build per-frame feature vector from normalized coords and confidences.
    Features per joint: [x_norm, y_norm, conf, vx, vy, angle_sum]
    coords_norm: (V,2)
    confs: (V,) visibility scores
    prev_coords_norm: (V,2) or None
    Returns: 1D feature vector of length V * 6
    """
    V = coords_norm.shape[0]

    # velocities in normalized coord space
    if prev_coords_norm is None:
        v = np.zeros_like(coords_norm)
    else:
        v = coords_norm - prev_coords_norm  # raw diff (vx, vy)
    
    # joint features
    joint_feats = []
    for j in range(V):
        x, y = coords_norm[j]
        conf = confs[j]
        vx, vy = v[j]
        joint_feats.extend([x, y, conf, vx, vy])

    # angle features
    angle_feats = []
    for (a, b, c) in ANGLE_TRIPLETS:
        ang = compute_angle(coords_norm[a], coords_norm[b], coords_norm[c])
        angle_feats.append(ang)

    return np.array(joint_feats + angle_feats, dtype=np.float32)

def build_sequences_from_triplets(
    frame_features, label,
    frames_per_second,
    duration_s=None,            # if None, infer from T and frames_per_second
    frames_per_sec_triplet=3,   # 3 frames per second in each sequence
    max_triplets_per_second=20, # K: cap on triplets per second
    num_global_sequences=5,     # M: how many sequences to sample per video
    min_gap=1,                  # minimal gap between frame indices in a triplet
    debug=True
):
    """
    frame_features: list or np.array of shape (T, D) per-frame features
    label: int class label for this video
    frames_per_second: p (how many frames you kept per second after sampling)
    duration_s: total duration in seconds (if None, inferred roughly as T // p)

    Returns:
        X_seqs: list of np.array, each shape (3*duration_s, D)
        y_seqs: list of labels
    """

    # ensure numpy array
    if frame_features.ndim != 2:
        raise ValueError(f"frame_features must be 2D (T,D), got shape {frame_features.shape}")

    T, D = frame_features.shape

    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be > 0")

    # infer duration if not provided
    if duration_s is None:
        duration_s = T // frames_per_second
        if duration_s == 0:
            if debug:
                print("[build_sequences_from_triplets] Not enough frames to infer duration.")
            return [], []

    if debug:
        print(f"[build_sequences_from_triplets] T={T}, D={D}, fps={frames_per_second}, "
              f"duration_s={duration_s}, frames_per_sec_triplet={frames_per_sec_triplet}")

    X_seqs = []
    y_seqs = []

    # 1) group frame indices by second (0..duration_s-1)
    sec_to_frames = {sec: [] for sec in range(duration_s)}
    for idx in range(T):
        sec = idx // frames_per_second
        if sec < duration_s:
            sec_to_frames[sec].append(idx)

    if debug:
        for sec in range(duration_s):
            print(f"  second {sec}: {len(sec_to_frames[sec])} frames -> indices {sec_to_frames[sec]}")

    # check we have enough frames in every second
    for sec in range(duration_s):
        n_frames = len(sec_to_frames[sec])
        if n_frames < frames_per_sec_triplet:
            if debug:
                print(f"[WARN] second {sec} has only {n_frames} frames; need {frames_per_sec_triplet}. Skipping video.")
            return [], []

    # 2) for each second, build candidate triplets and subsample them
    sec_to_triplets = {}
    for sec in range(duration_s):
        frame_ids = sec_to_frames[sec]  # e.g. [20,21,...,29] if p=10
        candidates = []
        # all combinations of 'frames_per_sec_triplet' frames
        for comb in combinations(frame_ids, frames_per_sec_triplet):
            # comb is a tuple like (i, j, k)
            # enforce minimal temporal gap if desired
            ok = True
            prev = comb[0]
            for idx2 in comb[1:]:
                if (idx2 - prev) < min_gap:
                    ok = False
                    break
                prev = idx2
            if ok:
                candidates.append(comb)

        if debug:
            print(f"  second {sec}: total candidate triplets before cap = {len(candidates)}")

        # cap candidates
        if len(candidates) > max_triplets_per_second:
            candidates = random.sample(candidates, max_triplets_per_second)
            if debug:
                print(f"  second {sec}: capped to {len(candidates)} triplets")

        sec_to_triplets[sec] = candidates

        if len(candidates) == 0:
            if debug:
                print(f"[WARN] second {sec} has no valid triplets after filtering. Skipping video.")
            return [], []

    # 3) build num_global_sequences sequences by picking 1 triplet per second
    for s in range(num_global_sequences):
        seq_frame_indices = []
        valid = True
        for sec in range(duration_s):
            triplets = sec_to_triplets[sec]
            if not triplets:
                valid = False
                if debug:
                    print(f"[WARN] no triplets in second {sec} for sequence {s}")
                break
            chosen_triplet = random.choice(triplets)
            seq_frame_indices.extend(chosen_triplet)

        if not valid:
            continue

        # sort indices to ensure chronological ordering across full video
        seq_frame_indices = sorted(seq_frame_indices)
        seq = frame_features[seq_frame_indices]  # shape (3*duration_s, D)
        X_seqs.append(seq)
        y_seqs.append(label)

        if debug and s == 0:
            print(f"  example sequence {s}: frame indices = {seq_frame_indices}")
            print(f"  example sequence shape: {seq.shape}")

    if debug:
        print(f"[build_sequences_from_triplets] generated {len(X_seqs)} sequences for this video.")

    return X_seqs, y_seqs

def build_lstm_samples_from_video(video_path, sample_fps=10, seq_len=3, max_frames=None):
    """
    Build LSTM-ready samples from a single video.

    - video_path: path to video file
    - label: numeric class label (e.g., 0 for pushup, 1 for pullup, etc.)
    - sample_fps: desired sampling rate (frames per second to process)
    - seq_len: sequence length in frames for LSTM (e.g., 3)
    - max_frames: optional cap to limit frames processed (for debugging)

    Returns:
        X_seqs: list of np.array of shape (seq_len, D)
        y_seqs: list of labels (same length as X_seqs)
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open:", video_path)
        return [], []

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 30.0  # fallback

    # compute frame step to approximate sample_fps
    step = max(1, int(round(orig_fps / sample_fps)))
    # print("################################################################################################")
    # print(f"[{video_path}] orig_fps={orig_fps:.2f}, sample_fps≈{orig_fps/step:.2f}, step={step}")

    frame_features = []
    frame_idx = 0
    prev_coords_norm = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # sample every "step" frames
        if frame_idx % step != 0:
            continue

        # optional limit
        if max_frames is not None and len(frame_features) >= max_frames:
            break

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if not results.pose_landmarks:
            # no pose detected -> skip this frame
            prev_coords_norm = None
            continue

        # collect coords & confs
        landmarks = results.pose_landmarks.landmark
        V = len(landmarks)
        coords = np.zeros((V, 2), dtype=np.float32)
        confs = np.zeros((V,), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            coords[i, 0] = lm.x
            coords[i, 1] = lm.y
            confs[i] = lm.visibility

        # normalize coords around hips + scale by shoulder distance
        coords_norm = normalize_coords(coords, mp_pose.POSE_CONNECTIONS)

        # build per-frame feature vector
        feat = build_frame_features(coords_norm, confs, prev_coords_norm)
        # print("--------------------------------------------------------------------------------")
        # print(feat)
        # print("--------------------------------------------------------------------------------")
        frame_features.append(feat)

        prev_coords_norm = coords_norm

    cap.release()
    cv2.destroyAllWindows()

    print(len(frame_features))

    # convert frame_features list to sequences of length seq_len
    frame_features = np.stack(frame_features, axis=0) if frame_features else np.empty((0,0))

    return frame_features

def build_fixed_length_sequences(
    frame_features,
    label,
    fps,
    target_len=100,
    num_sequences=5,
    debug=True
):
    """
    frame_features: (T, D) array of per-frame features for the full video
    label: int label for this video
    fps: sampling fps (e.g. 10)
    target_len: desired sequence length (e.g. 100 frames)
    num_sequences: how many different sequences to sample per video
    Returns:
        X_seqs: list of arrays (num_sequences, target_len, D)
        y_seqs: list of labels
    """

    frame_features = np.asarray(frame_features, dtype=np.float32)
    if frame_features.ndim != 2:
        raise ValueError(f"frame_features must be 2D (T,D), got {frame_features.shape}")
    T, D = frame_features.shape

    if T == 0:
        if debug:
            print("[build_fixed_length_sequences] Empty frame_features, skipping.")
        return [], []

    duration_s = T // fps
    if duration_s == 0:
        if debug:
            print("[build_fixed_length_sequences] Not enough frames for even 1 second, skipping.")
        return [], []

    # Only use full seconds
    usable_T = duration_s * fps
    frame_features = frame_features[:usable_T]
    T = usable_T

    if debug:
        print(f"[build_fixed_length_sequences] T={T}, D={D}, fps={fps}, duration_s={duration_s}")

    # If video longer than target_len seconds, you can clamp duration_s here
    if duration_s > target_len:
        if debug:
            print(f"[WARN] duration_s={duration_s} > target_len={target_len}, "
                  f"clamping to first {target_len} seconds.")
        duration_s = target_len
        usable_T = duration_s * fps
        frame_features = frame_features[:usable_T]
        T = usable_T

    # 1) group frame indices per second
    sec_to_frames = {sec: [] for sec in range(duration_s)}
    for idx in range(T):
        sec = idx // fps
        if sec < duration_s:
            sec_to_frames[sec].append(idx)

    if debug:
        for sec in range(duration_s):
            print(f"  second {sec}: {len(sec_to_frames[sec])} frames -> idx {sec_to_frames[sec]}")

    # 2) decide how many frames to pick from each second to reach target_len
    if target_len < duration_s:
        raise ValueError("target_len must be >= duration_s")

    base = target_len // duration_s
    extra = target_len % duration_s

    per_sec_counts = []
    for sec in range(duration_s):
        count = base + (1 if sec < extra else 0)
        per_sec_counts.append(count)

    if debug:
        print(f"  base={base}, extra={extra}, per_sec_counts={per_sec_counts}, sum={sum(per_sec_counts)}")

    X_seqs = []
    y_seqs = []

    # 3) build num_sequences sequences
    for s in range(num_sequences):
        chosen_indices = []
        for sec, count in enumerate(per_sec_counts):
            frame_ids = sec_to_frames[sec]
            L = len(frame_ids)

            if L == 0:
                # shouldn't happen if duration_s computed from T//fps
                continue

            if L >= count:
                # we can sample without replacement for more variation
                # for deterministic coverage, use linspace; for augmentation, use random.sample
                # here we'll mix: random choice but spread via linspace if you want
                # simplest: random.sample
                chosen = random.sample(frame_ids, count)
                chosen.sort()  # keep local chronological order
            else:
                # need to "stretch" frames: sample with replacement via linspace over indices
                pos = np.linspace(0, L - 1, count)
                idx_local = np.round(pos).astype(int)
                chosen = [frame_ids[i] for i in idx_local]

            chosen_indices.extend(chosen)

        # ensure global chronological order
        chosen_indices = sorted(chosen_indices)
        seq = frame_features[chosen_indices]  # shape (target_len, D)

        if debug and s == 0:
            print(f"  example sequence indices (seq 0): {chosen_indices[:30]} ... total={len(chosen_indices)}")
            print(f"  example sequence shape: {seq.shape}")

        X_seqs.append(seq)
        y_seqs.append(label)

    if debug:
        print(f"[build_fixed_length_sequences] generated {len(X_seqs)} sequences for this video.\n")

    return X_seqs, y_seqs

mp_pose = mp.solutions.pose
# Example: build angle triplets excluding face + hands as centers
EXCLUDED_CENTERS = list(range(0, 11)) + list(range(15, 23))  # tweak if needed
ANGLE_TRIPLETS = build_angle_triplets(mp_pose, exclude_centers=EXCLUDED_CENTERS)
num_angle_features = len(ANGLE_TRIPLETS)

videos_and_labels = [
    ("data/videos/bicep.mp4", 0),
    ("data/videos/pushup.mp4", 1),
    # ...
]
dataset_X = []
dataset_y = []
'''
sample_fps = 10  # p

raw_video_feats = []   # list of (T_i, D) arrays
raw_video_labels = []
durations_s = []       # per-video durations in seconds (integer)

for video_path, label in videos_and_labels:
    feats = build_lstm_samples_from_video(video_path, sample_fps=sample_fps)
    print(f"{video_path}: raw feats.shape = {feats.shape}")
    if feats.size == 0:
        continue

    T_i = feats.shape[0]
    duration_i = T_i // sample_fps   # integer number of "usable" seconds
    if duration_i < 1:
        print(f"Skipping {video_path}: too few frames.")
        continue

    raw_video_feats.append(feats)
    raw_video_labels.append(label)
    durations_s.append(duration_i)

common_duration_s = min(durations_s)   # e.g. 23 seconds
target_frames_per_video = common_duration_s * sample_fps
print("common_duration_s:", common_duration_s)
print("target_frames_per_video:", target_frames_per_video)

for feats, label in zip(raw_video_feats, raw_video_labels):
    # 1) resample per-video features to common length
    feats_resampled = resample_to_length(feats, target_frames_per_video)  # (target_frames_per_video, D)
    print(f"After resample: feats_resampled.shape = {feats_resampled.shape}")

    # 2) build per-video sequences using your triplet-based logic
    X_video, y_video = build_sequences_from_triplets(
        frame_features=feats_resampled,
        label=label,
        frames_per_second=sample_fps,      # p
        duration_s=common_duration_s,      # force same duration for all
        frames_per_sec_triplet=3,
        max_triplets_per_second=20,
        num_global_sequences=5,            # 5 sequences per video
        min_gap=1,
        debug=True                         # keep True while debugging
    )

    dataset_X.extend(X_video)  # each (3 * common_duration_s, D)
    dataset_y.extend(y_video)

'''

for video_path, label in videos_and_labels:
    # previously: frame_features -> build_sequences_from_triplets(...)
    # now:
    frame_features = build_lstm_samples_from_video(video_path, sample_fps=10)
    print(video_path, "raw frame_features:", frame_features.shape)

    X_video, y_video = build_fixed_length_sequences(
        frame_features=frame_features,
        label=label,
        fps=10,
        target_len=100,
        num_sequences=5,
        debug=True
    )

    dataset_X.extend(X_video)
    dataset_y.extend(y_video)

# 3) stack into final arrays
X = np.array(dataset_X, dtype=np.float32)  # shape (N_total_seqs, 3*common_duration_s, D)
y = np.array(dataset_y, dtype=np.int64)

print("Final X shape:", X.shape, "y shape:", y.shape)

seq_idx = 0
X_seq = X[seq_idx]    # shape (3,191)
y_seq = y[seq_idx]

print("X shape:", X.shape, "y shape:", y.shape)
print("Features:")
# print(X)
print(X_seq)
print(X_seq.shape[0])
print("Labels:")
# print(y)

def build_feature_names(num_joints=33, num_angle_features=26):
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

feature_names = build_feature_names(num_angle_features=num_angle_features)
print("Number of features:", len(feature_names))
print(feature_names[:10], "...")  # show first few names

dfs = []
for t in range(X_seq.shape[0]):
    row = X_seq[t]  # shape (191,)
    df = pd.DataFrame([row], columns=feature_names)
    df["frame"] = t
    df["label"] = y_seq
    dfs.append(df)

df_seq = pd.concat(dfs, axis=0)
print(df_seq.head(10))

def show_samples_per_label(X, y, feature_names, n=10):
    samples = []
    for seq, lbl in zip(X, y):
        # take first frame only for simplicity
        row = seq[0]
        df = pd.DataFrame([row], columns=feature_names)
        df["label"] = lbl
        samples.append(df)
    df_all = pd.concat(samples, axis=0)
    print(df_all.groupby("label").head(n))

show_samples_per_label(X, y, feature_names, n=3)

def show_top_sequences(X, y, feature_names, n=10):
    rows = []
    for i in range(min(n, len(X))):
        seq = X[i]
        lbl = y[i]
        for t, frame in enumerate(seq):
            df = pd.DataFrame([frame], columns=feature_names)
            df["frame"] = t
            df["seq"] = i
            df["label"] = lbl
            rows.append(df)
    return pd.concat(rows, axis=0)

df_preview = show_top_sequences(X, y, feature_names, n=10)
print(df_preview.head(20))
# ---- mediapipe setup ----
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

frame_idx = 0
sample_every = 1  # change to >1 if you want to skip frames, e.g., 3, 5...

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # optionally skip frames
    if frame_idx % sample_every != 0:
        continue

    color_change = frame_idx % 255
    # print(f"\n=== Frame #{frame_idx} ===")

    # BGR -> RGB for mediapipe
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    lmList = []
    if results.pose_landmarks:
        # draw skeleton
        mpDraw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mpPose.POSE_CONNECTIONS,
            landmark_drawing_spec=mpDraw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
            connection_drawing_spec=mpDraw.DrawingSpec(color=(color_change,255,0), thickness=2)
        )

        h, w, c = frame.shape
        # print keypoints
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

            # print(f"  kp {id:2d}: norm=({lm.x:.3f}, {lm.y:.3f}, {lm.z:.3f}), "
            #      f"pix=({cx:4d}, {cy:4d}), visibility={lm.visibility:.3f}")

        # example: mark a specific joint (id 14)
        if len(lmList) > 14:
            cv2.circle(frame, (lmList[14][1], lmList[14][2]), 8, (255-color_change, color_change, 255), cv2.FILLED)

    cv2.imshow("Video", frame)
    # press '1' to exit
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()

print("------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------")
print("\t\t\t\t Main Controller")
print("------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------")

# main.py

from feature_extractor import PoseFeatureExtractor
from sequence_builder import FixedLengthSequenceBuilder
from dataset_manager import VideoDataset

# 1) Define your videos & labels
videos_and_labels = [
    ("data/videos/bicep.mp4", 0),
    ("data/videos/pushup.mp4", 1),
    ("data/videos/pullup.mp4", 2),
    ("data/videos/squat2.mp4", 3),
    # add more...
]

label_names = ["bicep_curl", "pushup", "pullup", "squats"]  # same order as labels

# 2) Create components
feature_extractor = PoseFeatureExtractor(
    sample_fps=10,
    max_frames=None,
    exclude_angle_centers=None  # uses default exclusion (face+hands)
)

sequence_builder = FixedLengthSequenceBuilder(
    fps=10,
    target_len=100,
    num_sequences=100,
    debug=False
)

# 3) Build dataset
dataset = VideoDataset(feature_extractor, sequence_builder)
dataset.build_from_videos(videos_and_labels)

X = dataset.X       # shape (N, 100, D)
y = dataset.y       # shape (N,)
print("X shape:", X.shape, "y shape:", y.shape)

# 4) Inspect one sequence as DataFrame
df_seq0 = dataset.sequence_to_dataframe(seq_idx=0)
print(df_seq0.head(10))

# 5) Show sample per label
dataset.show_samples_per_label(n=3)

# 6) Show top sequences (first few frames) across sequences
# df_preview = dataset.show_top_sequences(n=3)

print("------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------")
print("\t\t\t\t Train and Test on LSTM")
print("------------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------------")

from LSTM import LSTMTrainer
from visualization import TrainingVisualizer

# 3) Create and train LSTM model
T, D = X.shape[1], X.shape[2]
num_classes = len(set(y))

trainer = LSTMTrainer(
    input_shape=(T, D),
    num_classes=num_classes,
    lstm_units=128,
    dropout_rate=0.3,
    learning_rate=1e-3
)

# 4) Train + get test split
history, test_metrics,  (X_test, y_test) = trainer.train_val_test(
    X, y,
    val_ratio=0.25,
    test_ratio=0.15,
    epochs=30,
    batch_size=16,
    print_every=2
)

# 5) Plot training curves
visualizer = TrainingVisualizer()
visualizer.plot_history(history, show=True, save_path=None)

# 6) Confusion matrix + classification report + per-class samples
cm, preds, probs = trainer.evaluate_with_confusion_and_report(
    X_test, y_test,
    label_names=label_names,
    samples_per_class=5
)

# 7) Plot confusion matrix
if cm is not None:
    visualizer.plot_confusion_matrix(
        cm,
        label_names=label_names,
        normalize=False,
        show=True,
        save_path=None
    )

# 8) Inference on a single sequence
# Example: use first sequence from dataset
seq = X[0]  # shape (100, D)
pred_idx, pred_proba, pred_name = trainer.predict_sequence(seq, label_names=label_names)

print(f"Predicted class index: {pred_idx}")
print(f"Predicted class name:  {pred_name}")
print(f"Predicted probability: {pred_proba:.4f}")
print(f"True label:            {y[0]} ({label_names[y[0]]})")

# 9) Example: predict on a single sequence
num_sample_test =  25
print("\n Prediction on Test Samples:")
def test_samples(X_test, ground_truth_recv, num_sample_test=25, is_ground_truth_same=0):
    correct_classified = 0
    for i in range(num_sample_test):
        X_test_len = len(X_test)
        random_idx = random.randint(0, X_test_len - 1)
        seq = X_test[random_idx]
        pred_class, pred_proba, pred_name = trainer.predict_sequence(seq, label_names=label_names)
        ground_truth = ground_truth_recv if (is_ground_truth_same == 1) else y_test[random_idx]
        if pred_class == ground_truth:
            correct_classified += 1
        print(f" [Test Number - {i:2}] Test ID:{random_idx:3} Predicted: {pred_class:2} ({pred_name:15}), Proba={pred_proba:.3f} Ground Truth: {ground_truth:2} ({label_names[ground_truth]:15})")
    print("\nTest Report:")
    print(f"  Correct classification: {correct_classified}")
    print(f"  Test Accuracy on Sample Data: {correct_classified/num_sample_test*100}%")

test_samples(X_test, 0, num_sample_test=num_sample_test)

def separator():
    print("-------------------------------------------------------------------------------------------------")

print("Testing on New Unseen Videos")
test_video_list = [
    ("data/videos/pushup2.mp4", 1),
    ("data/videos/bicep_curl_2.mp4", 0),
    ("data/videos/pullup2.mp4", 2),
    ("data/videos/pushup3.mp4", 1),
    ("data/videos/bicep_curl_3.mp4", 0),
    ("data/videos/pushup4.mp4", 1),
    ("data/videos/squat2.mp4", 3),
    ("data/videos/bicep_curl_4.mp4", 0),
    ("data/videos/pullup3.mp4", 2),
    ("data/videos/bicep_curl_5.mp4", 0),
    ("data/videos/pullup6.mp4", 2),
    ("data/videos/pullup7.mp4", 2),
    ("data/videos/squat3.mp4", 3),
    ("data/videos/Multiple_Equipment_Exercises.mp4", 2)
]
for video_path, ground_truth in test_video_list:
    separator()
    X_vid, _ = dataset.build_sequences_for_video(video_path)
    print(f"Video is getting tested on - {video_path}")
    print(len(X_vid), X_vid[0].shape)  # e.g. (num_sequences, 100, D)
    test_samples(X_vid, ground_truth, num_sample_test=num_sample_test, is_ground_truth_same=1)
