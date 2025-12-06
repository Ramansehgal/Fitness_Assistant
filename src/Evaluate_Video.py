import os, cv2
import mediapipe as mp

# ---- video setup ----
video_path = "data/videos/bicep.mp4"
cap = cv2.VideoCapture(video_path)

print("CWD:", os.getcwd())
print("Exists (video_path):", os.path.exists(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
print("Video FPS reported by OpenCV:", fps)

if not cap.isOpened():
    print("❌ Could not open video:", video_path)
    raise SystemExit


import cv2
import mediapipe as mp
import numpy as np
import math

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

def normalize_coords(coords):
    """
    coords: (V,2) array of MediaPipe normalized coords (0..1, 0..1)
    Returns root-relative, scale-normalized coords (V,2).
    Root = mid-hip (L_HIP, R_HIP).
    Scale = distance between shoulders.
    """
    root = (coords[L_HIP] + coords[R_HIP]) / 2.0
    coords_rel = coords - root

    # Scale by shoulder distance
    shoulder_dist = np.linalg.norm(coords[L_SH] - coords[R_SH])
    if shoulder_dist < 1e-6:
        shoulder_dist = 1.0
    coords_norm = coords_rel / shoulder_dist
    return coords_norm

def build_angle_triplets(exclude_centers=None):
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
        # if you want direction only, you could do:
        # v = np.sign(v)

    # angles per joint (we sum angles assigned to that joint as "middle" joint)
    angle_per_joint = np.zeros(V, dtype=np.float32)
    for (a,b,c) in ANGLE_TRIPLETS:
        ang = compute_angle(coords_norm[a], coords_norm[b], coords_norm[c])
        angle_per_joint[b] += ang

    features = []
    for j in range(V):
        x, y = coords_norm[j]
        conf = confs[j]
        vx, vy = v[j]
        ang = angle_per_joint[j]
        features.extend([x, y, conf, vx, vy, ang])

    return np.array(features, dtype=np.float32)  # shape (V*6,)

def build_lstm_samples_from_video(video_path, label, sample_fps=10, seq_len=3, max_frames=None):
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
    print(f"[{video_path}] orig_fps={orig_fps:.2f}, sample_fps≈{orig_fps/step:.2f}, step={step}")

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
        coords_norm = normalize_coords(coords)

        # build per-frame feature vector
        feat = build_frame_features(coords_norm, confs, prev_coords_norm)
        frame_features.append(feat)

        prev_coords_norm = coords_norm

    cap.release()
    cv2.destroyAllWindows()

    # convert frame_features list to sequences of length seq_len
    frame_features = np.stack(frame_features, axis=0) if frame_features else np.empty((0,0))
    X_seqs = []
    y_seqs = []

    if frame_features.shape[0] >= seq_len:
        for start in range(0, frame_features.shape[0] - seq_len + 1):
            seq = frame_features[start:start+seq_len]  # shape (seq_len, D)
            X_seqs.append(seq)
            y_seqs.append(label)

    return X_seqs, y_seqs

# Example: build angle triplets excluding face + hands as centers
EXCLUDED_CENTERS = list(range(0, 11)) + list(range(15, 23))  # tweak if needed
ANGLE_TRIPLETS = build_angle_triplets(exclude_centers=EXCLUDED_CENTERS)

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
    print(f"\n=== Frame #{frame_idx} ===")

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

            print(f"  kp {id:2d}: norm=({lm.x:.3f}, {lm.y:.3f}, {lm.z:.3f}), "
                  f"pix=({cx:4d}, {cy:4d}), visibility={lm.visibility:.3f}")

        # example: mark a specific joint (id 14)
        if len(lmList) > 14:
            cv2.circle(frame, (lmList[14][1], lmList[14][2]), 8, (255-color_change, color_change, 255), cv2.FILLED)

    cv2.imshow("Video", frame)
    # press '1' to exit
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()
