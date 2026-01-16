import os
import cv2
import math
import time
import random
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import defaultdict
from itertools import combinations


import poseestimation as pm



# ---- video setup ----
video_path = "data/videos/bicep_curl_4.mp4"
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

from rich.console import Console
from rich.table import Table
import pandas as pd

def display_dataframe_rich(
    df: pd.DataFrame,
    max_rows: int = 10,
    max_cols: int = None,
    title: str = "DataFrame Preview"
):
    console = Console()

    if max_rows:
        df = df.head(max_rows)
    if max_cols:
        df = df.iloc[:, :max_cols]

    table = Table(title=title, show_lines=True)

    for col in df.columns:
        table.add_column(str(col), justify="right", overflow="fold")

    for _, row in df.iterrows():
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])

    console.print(table)

def display_dataframe_html(df, filename="df_preview.html"):
    html = df.to_html(border=1)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"DataFrame rendered to {filename} (open in browser)")

import pandas as pd
import numpy as np

def print_dataset_summary(df):
    keypoint_cols = [c for c in df.columns if c.startswith("j")]
    angle_cols = [c for c in df.columns if c.startswith("angle")]
    meta_cols = [c for c in df.columns if c not in keypoint_cols + angle_cols]

    print("\n\t\t Each Video Dataset Summary")
    print("=" * 60)
    print(f"Rows                : {df.shape[0]}")
    print(f"Total Columns       : {df.shape[1]}")
    print(f"Keypoint features   : {len(keypoint_cols)}")
    print(f"Angle features      : {len(angle_cols)}")
    print("=" * 60)



def select_representative_columns(
    df,
    first_kpts=1,
    last_kpts=1,
    angle_tail=5
):
    # Keypoints grouped by joint index
    def kp_group(j):
        return [c for c in df.columns if c.startswith(f"j{j:02d}_")]

    kp_indices = sorted({int(c[1:3]) for c in df.columns if c.startswith("j")})
    
    first_joints = kp_indices[:first_kpts]
    last_joints = kp_indices[-last_kpts:]

    cols = []
    for j in first_joints:
        cols.extend(kp_group(j))

    for j in last_joints:
        cols.extend(kp_group(j))

    angle_cols = [c for c in df.columns if c.startswith("angle")]
    meta_cols = [c for c in df.columns if not c.startswith(("j", "angle"))]

    cols.extend(angle_cols[-angle_tail:])
    cols.extend(meta_cols)

    # Remove duplicates, preserve order
    seen = set()
    cols = [c for c in cols if not (c in seen or seen.add(c))]
    return cols


from rich.console import Console
from rich.table import Table

def display_dataframe_rich_smart(
    df,
    max_rows=5,
    title="Dataset Preview (Smart View)"
):
    console = Console()

    print_dataset_summary(df)

    cols = select_representative_columns(df)
    df_view = df[cols].head(max_rows)

    table = Table(title=title, show_lines=True, header_style="bold cyan")

    for c in df_view.columns:
        style = "green" if c.startswith("j") else "magenta" if c.startswith("angle") else "yellow"
        table.add_column(c, justify="right", style=style, overflow="fold")

    for _, row in df_view.iterrows():
        formatted_row = []
        for col, v in zip(df_view.columns, row):
            if col.startswith(("j", "angle")):
                # Pose & angle features → float
                formatted_row.append(f"{float(v):.3f}")
            else:
                # Meta columns → int
                formatted_row.append(str(int(v)))
        table.add_row(*formatted_row)

    console.print(table)

def display_dataframe_html_smart(
    df,
    filename="df_preview.html",
    max_rows=10
):
    cols = select_representative_columns(df)
    df_view = df[cols].head(max_rows)

    def color_col(c):
        if c.startswith("j"):
            return "background-color:#e8f5e9"
        elif c.startswith("angle"):
            return "background-color:#e3f2fd"
        else:
            return "background-color:#fff3e0"

    styles = [
        dict(
            selector="th",
            props=[
                ("position", "sticky"),
                ("top", "0"),
                ("background-color", "#263238"),
                ("color", "white"),
                ("font-size", "12px"),
                ("padding", "6px"),
            ],
        ),
        dict(
            selector="td",
            props=[
                ("padding", "6px"),
                ("font-size", "11px"),
                ("white-space", "nowrap"),
            ],
        ),
        dict(
            selector="table",
            props=[
                ("border-collapse", "collapse"),
                ("width", "100%"),
            ],
        ),
    ]

    styled = (
        df_view.style
        .set_table_styles(styles)
        .applymap(lambda _: "border:1px solid #bbb")
        .applymap(color_col)
        .format(precision=4)
    )

    html = f"""
    <html>
    <head>
    <style>
    body {{ font-family: Arial; }}
    .container {{
        overflow-x: auto;
        max-width: 100%;
    }}
    </style>
    </head>
    <body>
    <h3>Dataset Preview (Smart View)</h3>
    <div class="container">
    {styled.to_html()}
    </div>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"HTML preview written to: {filename}")

def pretty_print_dataframe(
    df: pd.DataFrame,
    max_rows: int = 10,
    max_cols: int = 15,
    float_precision: int = 4,
    transpose: bool = False,
    title: str = None
):
    """
    Print a pandas DataFrame in a clean, readable tabular format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    max_rows : int
        Max number of rows to display
    max_cols : int
        Max number of columns to display
    float_precision : int
        Decimal precision for float values
    transpose : bool
        If True, prints DataFrame transposed (useful for many features)
    title : str
        Optional title for clarity
    """

    if title:
        print(f"\n{'=' * len(title)}")
        print(title)
        print(f"{'=' * len(title)}")

    pd.set_option("display.max_rows", max_rows)
    pd.set_option("display.max_columns", max_cols)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.{float_precision}f}")

    if transpose:
        df = df.T

    display_df = df.copy()

    # Reset index for clean display
    if display_df.index.name or isinstance(display_df.index, pd.MultiIndex):
        display_df = display_df.reset_index()

    print(display_df.to_string(index=False))

    # Restore defaults (important for notebooks)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.float_format")


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
print("Line:681")
display_dataframe_rich(df_seq, max_rows=5, max_cols=10, title="Frame Features")
# display_dataframe_html(df_seq, filename="Frame_Features.html")
display_dataframe_rich_smart(df_seq, max_rows=5, title="Frame Features")
# display_dataframe_html_smart(df_seq, filename="Frame_Features.html")
# pretty_print_dataframe(df_seq.head(1),transpose=True,title="Frame Feature Breakdown")
print("Line:683")

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
    print("Line:695")
    display_dataframe_rich(df_all, max_rows=5, max_cols=10, title="Each Label Frame Sequences")
    # display_dataframe_html(df_seq, filename="Per_Class_Sequences.html")
    display_dataframe_rich_smart(df_all, max_rows=5, title="Per Class Sample Frame Dataset")
    # display_dataframe_html_smart(df_seq, filename="Frame_Features.html")
    # pretty_print_dataframe(df_seq, title="Per_Class_Sequences")
    print("Line:697")

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
print("Line:716")
display_dataframe_rich(df_preview, max_rows=5, max_cols=10, title="10 Sample Frame Sequences")
# display_dataframe_html(df_seq, filename="Sample_Sequences.html")
display_dataframe_rich_smart(df_seq, max_rows=5, title="Frame Features")
# display_dataframe_html_smart(df_seq, filename="Frame_Features.html")
# pretty_print_dataframe(df_seq, title="Sample_Sequences")
print("Line:718")

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
    ("data/videos/bicep_curl_4.mp4", 0),
    ("data/videos/bicep_curl_6.mp4", 0),
    ("data/videos/bicep_Curl5.mp4", 0),
    ("data/videos/pushup.mp4", 1),
    ("data/videos/pushup3.mp4", 1),
    ("data/videos/pushup7.mp4", 1),
    ("data/videos/pullup.mp4", 2),
    ("data/videos/pullup4.mp4", 2),
    ("data/videos/pullup5.mp4", 2),
    ("data/videos/squat.mp4", 3),
    ("data/videos/squat2.mp4", 3),
    ("data/videos/squat_4.mp4", 3),
    ("data/videos/squat5.mp4", 3),
    # add more...
]

test_video_list = [
    ("data/videos/pushup2.mp4", 1),
    ("data/videos/bicep_curl_2.mp4", 0),
    ("data/videos/pullup2.mp4", 2),
    ("data/videos/squat3.mp4", 3),
    ("data/videos/pullup6.mp4", 2),
    ("data/videos/pushup_4.mp4", 1),
    ("data/videos/bicep_curl_3.mp4", 0),
    ("data/videos/pushup5.mp4", 1),
    ("data/videos/squat_6.mp4", 3),
    ("data/videos/bicep_curl_7.mp4", 0),
    ("data/videos/pullup3.mp4", 2),
    ("data/videos/bicep_curl_8.mp4", 0),
    ("data/videos/pullup7.mp4", 2),
    ("data/videos/squat_7.mp4", 3),
    ("data/videos/bicep_curl9.mp4", 0),
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

def dataset_info(arr1, arr2, msg=""):
    # ---------------------------
    # Step 3: Append arrays as new columns
    # ---------------------------
    # combined = np.concatenate((arr1, arr2), axis=0)

    # Create DataFrame
    df = pd.DataFrame(arr1)
    df['label'] = arr2

    print("DataFrame after appending arrays:\n", df.head(), "\n")

    # ---------------------------
    # Step 4: Identify columns where second array has unique values
    # ---------------------------
    unique_values = df['label'].unique()
    print("Unique values in num_col2:", unique_values, "\n")

    # ---------------------------
    # Step 5: Filter top 5 rows for each unique value
    # (Sorting by 'score' in descending order)
    # ---------------------------
    top5_each = (
        df.sort_values(by='score', ascending=False)
        .groupby('num_col2')
        .head(4)
        .reset_index(drop=True)
    )

    display_dataframe_rich_smart(top5_each, max_rows=10, title=msg)
    display_dataframe_rich_smart(top5_each, max_rows=10, title="Per Class Sample Frame Dataset")
    print("Top 5 rows for each unique value in num_col2:\n", top5_each)


# 3) Build dataset
dataset = VideoDataset(feature_extractor, sequence_builder)
dataset.build_from_videos(videos_and_labels)

X = dataset.X       # shape (N, 100, D)
y = dataset.y       # shape (N,)
print("Training Dataset Shape")
print("=" * 60)
print("X shape:", X.shape, "\ny shape:", y.shape)
print("=" * 60)

# 4) Inspect one sequence as DataFrame
df_seq0 = dataset.sequence_to_dataframe(seq_idx=0)
print(df_seq0.head(10))
display_dataframe_rich_smart(df_seq0, title="Sample Sequence From DataFrame")

# 5) Show sample per label
dataset.show_samples_per_label(n=3)

# dataset_info(X,y,msg="Training Dataset")

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
correct_class_pred = [0] * 4
incorrect_class_pred = [0] * 4
accuracy_per_class = [0.00] * 4
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
    incorrect_classified = num_sample_test - correct_classified
    if is_ground_truth_same == 1:
        correct_class_pred[ground_truth_recv] += correct_classified
        incorrect_class_pred[ground_truth_recv] += incorrect_classified
        accuracy_per_class[ground_truth_recv] = (accuracy_per_class[ground_truth_recv] * 100 + correct_classified)/(100 + num_sample_test)
    print("\nTest Report:")
    print(f"  Correct classification: {correct_classified}")
    print(f"  Test Accuracy on Sample Data: {correct_classified/num_sample_test*100}%")

test_samples(X_test, 0, num_sample_test=num_sample_test)

def separator():
    print("-------------------------------------------------------------------------------------------------")

print("Testing on New Unseen Videos")
print("=" * 60)
print(f"Number of Test Videos               : {len(test_video_list)}")
print(f"Number of Sample Sequence Per Video : {num_sample_test}")
print("=" * 60)

for video_path, ground_truth in test_video_list:
    separator()
    X_vid, _ = dataset.build_sequences_for_video(video_path)
    print(f"Video is getting tested on - {video_path}")
    print(len(X_vid), X_vid[0].shape)  # e.g. (num_sequences, 100, D)
    test_samples(X_vid, ground_truth, num_sample_test=num_sample_test, is_ground_truth_same=1)

total_correct = sum(correct_class_pred)
toal_test_cases = num_sample_test * len(test_video_list)
overall_accuracy = total_correct/toal_test_cases

print(f"Each classes correct predition count   : {correct_class_pred}")
print(f"Each classes incorrect predition count : {correct_class_pred}")
print(f"Each classes prediction accuracy       : {correct_class_pred}")
print(f"Total Correct Prediction               : {total_correct}")
print(f"Total Test Sequences                   : {toal_test_cases}")
print(f"Overall Model Accuracy                 : {overall_accuracy*100}%")


from video_segmentation import segment_and_plot_timeline

segment_video_list = [
    # ("data/videos/squat3.mp4", 3),
    # ("data/videos/pullup6.mp4", 2),
    # ("data/videos/pushup_4.mp4", 1),
    # ("data/videos/bicep_curl_3.mp4", 0),
    # ("data/videos/pushup5.mp4", 1),
    # ("data/videos/squat_6.mp4", 3),
    ("data/videos/bicep_curl_7.mp4", 0),
    ("data/videos/pullup3.mp4", 2),
    ("data/videos/bicep_curl_8.mp4", 0),
    ("data/videos/pullup7.mp4", 2),
    ("data/videos/Multiple_Equipment_Exercises.mp4",0),
    ("data/videos/merged_video.mp4",0)
]

print(trainer)

for video_path, label in segment_video_list:

    segments, window_preds = segment_and_plot_timeline(
        video_path=video_path,
        trainer=trainer,
        feature_extractor=feature_extractor,
        label_names=label_names,
        window_size=50,                      # 100 frames, same as training
        stride_frames=10,                    # slide ~20 frames each step
        min_segment_windows=1                # ignore very tiny segments
    )


#---------------------------------------------------------------------------------------------------------------#
#                                       Display Text and Rep Count on the video
#---------------------------------------------------------------------------------------------------------------#
import poseestimation as pm
'''
cap = cv2.VideoCapture("data/videos/bicep.mp4")
detector = pm.poseDetector()
dir = 0
count=0
ptime = 0
frame_id = 0
def compute_angle2(a, b, c):
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

while True:
    ret, frame = cap.read()
    print(f"\n=== Frame #{frame_id} ===")
    frame_id = frame_id + 1
    if ret:
        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame)
        #print(lmList)
        if len(lmList)!=0:
            angle=detector.findAngle(frame, 12, 14, 16, draw=True)
            _ , cos_a1, cos_a2 = lmList[12]
            _ , cos_b1, cos_b2 = lmList[14]
            _ , cos_c1, cos_c2 = lmList[16]
            cos_a = np.array([cos_a1, cos_a2], dtype=np.float32)
            cos_b = np.array([cos_b1, cos_b2], dtype=np.float32)
            cos_c = np.array([cos_c1, cos_c2], dtype=np.float32)
            cos_angle = compute_angle2(cos_a, cos_b, cos_c) * 180 / 3.14
            print(f"Cos Angle Between Joints a:{cos_a} b:{cos_b} c:{cos_c} = {cos_angle}")
            #print(angle)
            per=np.interp(angle, (190,300), (0,100))
            bar = np.interp(angle, (190, 300), (100, 650))
            color=(255, 100, 100)
            if per == 100:
                color=(100, 255, 100)
                if dir==0:
                    count+=0.5
                    dir=1
            if per == 0:
                color=(100, 100, 255)
                if dir == 1:
                    count+=0.5
                    dir=0

            # Displaying Curl Count
            pos = [30, 450]
            ox, oy = pos[0], pos[1]
            offset = 10
            text = str(int(count))

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 11, 11)
            x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, text, (ox, oy), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 6)

            #Displating the Bar Count

            cv2.rectangle(frame, (1100, 100), (1175, 650),color, 3)
            cv2.rectangle(frame, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(frame, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)


            # Displaying the FPS
            ctime = time.time()
            fps = 1/(ctime - ptime)
            ptime = ctime
            pos = [30, 60]
            ox, oy = pos[0], pos[1]
            offset=10
            text = "FPS: " + str(int(fps))

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 3,3)
            x1,y1, x2, y2 = ox-offset, oy+offset, ox+w+offset, oy-h-offset
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, text, (ox, oy), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

            print(f"FPS:{fps} BAR:{bar} Percentage Complete:{per} Rep Count:{count}")

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break
    else:
        break


cap = cv2.VideoCapture("data/videos/pushup.mp4")
detector = pm.poseDetector()
dir = 0
count=0
ptime = 0
while True:
    ret, frame = cap.read()
    if ret:
        frame = detector.findPose(frame)
        lmList = detector.findPosition(frame, draw=False)
        #print(lmList)
        if len(lmList)!=0:
            angle=detector.findAngle(frame, 11, 13, 15, draw=True)
            #print(angle)
            per=np.interp(angle, (200,280), (0,100))
            bar = np.interp(angle, (200, 280), (650, 100))
            color=(255, 100, 100)
            if per == 100:
                color=(100, 255, 100)
                if dir==0:
                    count+=0.5
                    dir=1
            if per == 0:
                color=(100, 100, 255)
                if dir == 1:
                    count+=0.5
                    dir=0

            # Displaying Curl Count
            pos = [30, 450]
            ox, oy = pos[0], pos[1]
            offset = 10
            text = str(int(count))

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 11, 11)
            x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, text, (ox, oy), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 6)

            #Displating the Bar Count

            cv2.rectangle(frame, (1600, 100), (1675, 650),color, 3)
            cv2.rectangle(frame, (1600, int(bar)), (1675, 650), color, cv2.FILLED)
            cv2.putText(frame, f'{int(per)}%', (1600, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)


            # Displaying the FPS
            ctime = time.time()
            fps = 1/(ctime - ptime)
            ptime = ctime
            pos = [30, 60]
            ox, oy = pos[0], pos[1]
            offset=10
            text = "FPS: " + str(int(fps))

            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 3,3)
            x1,y1, x2, y2 = ox-offset, oy+offset, ox+w+offset, oy-h-offset
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), cv2.FILLED)
            cv2.putText(frame, text, (ox, oy), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        frame = cv2.resize(frame, (0,0), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF==ord('1'):
            break
    else:
        break

'''

#----------------------------------------------------------------------------
'''
Internal Videos
V1 :    10s --> 100F 10fps, 30fps
        1s : 10 frames out of 30   [F1_1, F1_2, ... F1_30] -> 3OC10
        Examples -  E0 : 100 F -> Pushup        {Train}
                    E1 : 100 F -> Pushup        {Train}
                    E2 : 100 F -> Pushup        {Test} [Internal Test]

V2 :

External Videos -
'''
#----------------------------------------------------------------------------