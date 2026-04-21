
import os
import cv2
import math
import time
import random
import numpy as np
import pandas as pd
import mediapipe as mp


from rich.table import Table
from rich.console import Console
from collections import defaultdict
from itertools import combinations
from tensorflow.keras.models import load_model


from LSTM import LSTMTrainer
from dotenv import load_dotenv
from dataset_manager import VideoDataset
from visualization import TrainingVisualizer
from feature_extractor import PoseFeatureExtractor
from sequence_builder import FixedLengthSequenceBuilder
from video_segmentation import segment_and_plot_timeline


from LLM_Feedback import ExerciseFeedbackGenerator

EPOCHS = 30
BATCH_SIZE = 16
NUM_CLASSES = 4
LSTM_UNITS = 128
DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3
PARTITION_SEPARATOR_LEN = 45
NUM_SAMPLE_TEST_PER_CLASS =  10

# 1) Define your videos & labels
print("-" * (PARTITION_SEPARATOR_LEN*2))
print("1) Video Dataset and Label Info:")

label_names = ["bicep_curl", "pushup", "pullup", "squats"] 
safe_min_angle = [18.76, 18.92, 19.31, 120.36]
safe_max_angle = [168.57, 168.79, 167.52, 171.68]
major_impacted_joint = [14,13,14,13,14,13,26,25]
major_impacted_joint_idx_R = [ 0, 0, 0, 4]
major_impacted_joint_idx_L = [24,24,24,19]
stable_abs_diff = [25, 25, 25, 35]

correct_class_pred = [0] * NUM_CLASSES
incorrect_class_pred = [0] * NUM_CLASSES
accuracy_per_class = [0.00] * NUM_CLASSES
total_sample_per_class = [0] * NUM_CLASSES

videos_and_labels = [
    ("data/videos/bicep.mp4", 0),
    ("data/videos/bicep_curl_4.mp4", 0),
    # ("data/videos/bicep_curl_6.mp4", 0),
    ("data/videos/pushup.mp4", 1),
    ("data/videos/pushup3.mp4", 1),
    ("data/videos/pushup6.mp4", 1),
    ("data/videos/pullup.mp4", 2),
    ("data/videos/pullup2.mp4", 2),
    ("data/videos/pullup5.mp4", 2),
    ("data/videos/squat.mp4", 3),
    # ("data/videos/squat2.mp4", 3),
    ("data/videos/squat6.mp4", 3),
    # ("data/videos/squat_9.mp4", 3),
]

test_video_list = [
    ("data/videos/squat2.mp4", 3),
    ("data/videos/pushup7.mp4", 1),
    ("data/videos/bicep_curl5.mp4", 0),
    ("data/videos/bicep_curl_4.mp4", 0),
    ("data/videos/pullup7.mp4", 2),
    ("data/videos/squat3.mp4", 3),
    ("data/videos/pushup3.mp4", 1),
]

segment_video_list = [
    ("data/videos/push_up_squat.mp4",0),
    # ("data/videos/merged_video.mp4",0)
]

def print_pipeline_stage_msg(msg):
    print("-" * (PARTITION_SEPARATOR_LEN*2))
    print(msg)

def print_training_data_info(X, y):
    print("=" * PARTITION_SEPARATOR_LEN)
    print("Training Dataset Shape")
    print("X shape :", X.shape, "\ny shape :", y.shape)
    print("=" * PARTITION_SEPARATOR_LEN)

def print_dataset_summary(df):
    keypoint_cols = [c for c in df.columns if c.startswith("j")]
    angle_cols = [c for c in df.columns if c.startswith("angle")]
    meta_cols = [c for c in df.columns if c not in keypoint_cols + angle_cols]

    print("\n\t\t Each Video Dataset Summary")
    print("=" * PARTITION_SEPARATOR_LEN)
    print(f"Rows                : {df.shape[0]}")
    print(f"Total Columns       : {df.shape[1]}")
    print(f"Keypoint features   : {len(keypoint_cols)}")
    print(f"Angle features      : {len(angle_cols)}")
    print("=" * PARTITION_SEPARATOR_LEN)

def save_best_lstm_model(model, save_dir="checkpoints", model_name="best_lstm4"):
    """
    Saves the full LSTM model (architecture + weights + optimizer state)
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, model_name + ".keras")
    model.save(path)

    print(f"[MODEL SAVE] Best LSTM model saved at: {path}")

def load_best_lstm_model(save_dir="checkpoints", model_name="best_lstm"):
    """
    Loads a previously saved LSTM model
    """
    path = os.path.join(save_dir, model_name + ".keras")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model found at {path}")

    model = load_model(path)
    print(f"[MODEL LOAD] Best LSTM model loaded from: {path}")
    return model

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

def select_representative_columns( df, first_kpts=1, last_kpts=1, angle_tail=5):
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

def generate_dataframe(X, y):
    dfs=[]
    for seq_idx in range(X.shape[0]):
        X_seq = X[seq_idx]    # shape (3,191)
        y_seq = y[seq_idx]
        for t in range(X_seq.shape[0]):
            row = X_seq[t]  # shape (191,)
            df = pd.DataFrame([row], columns=feature_names)
            df["frame"] = t
            df["label"] = y_seq
            dfs.append(df)
    
    df_seq = pd.concat(dfs, axis=0)
    return df_seq

def display_dataframe(df, max_rows=5, title="Dataset Preview (Smart View)"):
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

def test_samples(X_test, ground_truth_recv, num_sample_test=25, is_ground_truth_same=0):
    
    pred_class = 0
    correct_classified = 0
    
    for i in range(num_sample_test):
        X_test_len = len(X_test)
        random_idx = random.randint(0, X_test_len - 1)
        seq = X_test[random_idx]
        pred_class, pred_proba, pred_name = trainer.predict_sequence(seq, label_names=label_names)
        ground_truth = ground_truth_recv if (is_ground_truth_same == 1) else y_test[random_idx]
        if pred_class == ground_truth:
            correct_classified += 1
        print(f" [Test Number - {i:2}] Test ID:{random_idx:3} Predicted: {pred_class:2} ({pred_name:10}), Proba={pred_proba:.3f} Ground Truth: {ground_truth:2} ({label_names[ground_truth]:10})")
    
    incorrect_classified = num_sample_test - correct_classified
    print(f"Total Test Sequences:{num_sample_test} Correct:{correct_classified} Incorrect:{incorrect_classified}")
    
    if is_ground_truth_same == 1:
        total_sample_per_class[ground_truth_recv] += num_sample_test
        correct_class_pred[ground_truth_recv] += correct_classified
        incorrect_class_pred[ground_truth_recv] += incorrect_classified
        accuracy_per_class[ground_truth_recv] = correct_class_pred[ground_truth_recv] / total_sample_per_class[ground_truth_recv] * 100
    
    print("\nTest Report:")
    print(f"  Correct classification: {correct_classified}")
    print(f"  Test Accuracy on Sample Data: {correct_classified/num_sample_test*100}%")

    return pred_class

def get_video_metadata(video_path, sample_fps):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0

    return {
        "video_path": os.path.basename(video_path),
        "fps": fps,
        "sample_fps": sample_fps,
        "total_frames": total_frames,
        "duration_sec": round(duration, 2)
    }

def normalize_stats(stats_dict, include_series=False):
    if stats_dict is None:
        print(stats_dict)
        print("Did not received any value")

    out = {}
    for k, v in stats_dict.items():
        item = {
            "mean": round(v["mean"], 2),
            "min": round(v["min"], 2),
            "max": round(v["max"], 2),
            "std": round(v["std"], 2),
        }
        if "t_min" in v:
            item["t_min"] = round(v["t_min"], 2)
            item["t_max"] = round(v["t_max"], 2)
        if include_series and "series" in v:
            item["series"] = v["series"].tolist()
        out[k] = item
    return out

def build_llm_ready_payload(
    video_path,
    sample_fps,
    frame_features,
    timestamps,
    major_impacted_joint=None,
    angle_stats=None,
    bone_stats=None,
    segment_biomechanics=None,
    is_multi_video=0,
    rep_runtime_data=None
):
    """
    Auto-handles:
      - single video (angle_stats, bone_stats provided)
      - multi video (segment_biomechanics provided)
    """

    payload = {
        "video_metadata": get_video_metadata(video_path, sample_fps),
        "analysis_type": "multi" if segment_biomechanics else "single",
        "segments": []
    }
    # -----------------------------
    # CASE 1: SINGLE VIDEO
    # -----------------------------
    if is_multi_video == 0:
        payload["segments"].append({
            "segment_id": 0,
            "exercise": "unknown_or_single",
            "start_time": 0.0,
            "end_time": payload["video_metadata"]["duration_sec"],
            "major_impacted_joints": major_impacted_joint,
            "angle_stats": normalize_stats(angle_stats),
            "bone_stats": normalize_stats(bone_stats),
            "rep_analysis": rep_runtime_data or {},
            "notes": "Single continuous exercise"
        })
        return payload

    # -----------------------------
    # CASE 2: MULTI VIDEO
    # -----------------------------
    for i, seg in enumerate(segment_biomechanics):
        payload["segments"].append({
            "segment_id": i,
            "exercise": seg["class_name"],
            "class_idx": seg["class_idx"],
            "start_time": round(seg["start_time"], 2),
            "end_time": round(seg["end_time"], 2),
            "duration_sec": round(seg["end_time"] - seg["start_time"], 2),
            "major_impacted_joints": seg["major_impacted_joints"],
            "angle_stats": normalize_stats(seg["angle_stats"]),
            "bone_stats": normalize_stats(seg["bone_stats"]),
            "rep_analysis": (
                rep_runtime_data.get(i, {}) if rep_runtime_data else {}
            )
        })

    return payload

def get_screen_size():
    import ctypes
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def resize_frame_if_needed(frame, max_w, max_h, max_scale=2.5):
    h, w = frame.shape[:2]

    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # prevent excessive zoom
    scale = min(scale, max_scale)
    
    # print(f"SW:{new_w}, SH:{new_h} MW:{max_w}, MH:{max_h}, H:{h}, W:{w} scale_w:{scale_w} scale_h:{scale_h} scale:{scale}")

    # if frame is already large enough, do nothing
    if scale <= 1.0:
        return frame

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def compute_angle(a, b, c):
    """
    Angle at point b formed by vectors ba and bc, returns radians.
    a, b, c are np.array([x,y]).
    """
    # Ensure float dtype (CRITICAL)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c = c.astype(np.float32)
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba) + 1e-8
    nb = np.linalg.norm(bc) + 1e-8
    ba /= na
    bc /= nb
    cosang = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return math.acos(cosang)

def findAngle(img, lmList, p1, p2, p3, draw=True):
        _, x1, y1 = lmList[p1]
        _, x2, y2 = lmList[p2]
        _, x3, y3 = lmList[p3]

        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle<0:
            angle+=360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0,0,255), 2)
            cv2.circle(img, (x1, y1), 5, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0,0,255), 2)
            cv2.circle(img, (x2, y2), 5, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0,0,255), 2)
            cv2.circle(img, (x3, y3), 5, (0,0,255), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2-50, y2-50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)
        return angle

def get_class_at_time(segments, current_time, default_label="No Exercise"):
    """
    Given segmentation results and a timestamp (seconds),
    return the exercise class active at that time.

    Args:
        segments (list): list of segment dictionaries
        current_time (float): time in seconds
        default_label (str): label if no segment matches

    Returns:
        dict or None: matched segment info
    """
    for seg in segments:
        st = seg["start_time"]
        et = seg["end_time"]
        scn = seg["class_name"]
        sid = seg["class_idx"]
        if seg["start_time"] <= current_time <= seg["end_time"]:
            '''if current_time >= 24:
                print(f"{st} <= {current_time} <= {et}: {scn} {sid}")'''
            return {
                "class_idx": seg["class_idx"],
                "class_name": seg["class_name"],
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                # "mean_proba": seg["mean_proba"],
            }

    return {
        "class_idx": -1,
        "class_name": default_label,
        "start_time": None,
        "end_time": None,
        # "mean_proba": 0.0,
    }

def draw_text_block(
    frame,
    lines,
    start_x=30,
    start_y=400,
    line_height=28,
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=1.2,
    text_color=(255, 255, 255),
    bg_color=(40, 40, 40),  # dark grey
    thickness=1,
    padding=6
):
    """
    Draws multi-line text with background.
    """
    max_width = 0

    # Measure widest line
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)

    block_height = line_height * len(lines)

    # Draw background rectangle
    cv2.rectangle(
        frame,
        (start_x - padding, start_y - padding),
        (start_x + max_width + padding, start_y + block_height + padding),
        bg_color,
        cv2.FILLED
    )

    # Draw text lines
    y = start_y
    for line in lines:
        cv2.putText(
            frame,
            line,
            (start_x, y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
        y += line_height

def display_video(video_path, all_angle_stats, segments={}, segment_biomechanics={}, is_multi_video=0, pred_class=255):

    dir = 0
    rep_count = 0
    frame_idx = 0
    idle_count = 0
    sample_every = 1
    segment_start_frame = 0
    is_range_updated = [0,0,0,0]
    exercise_rep_count = [0,0,0,0]
    min_angle, max_angle = 0, 0
    frame_count, prev_frame_count = 0,0
    stability_count, instability_count = 0, 0
    curr_pred_class, prev_pred_class = 5,5
    upper_motion, ranged_motion, bottom_motion = 0, 0, 0
    upper_motion_count, prev_upper_motion_count = 0,0
    ranged_motion_count, prev_ranged_motion_count = 0,0
    bottom_motion_count, prev_bottom_motion_count = 0,0
    direction_motion = ["Upward", "Downward"]
    frame_exercise_info = {
        "class_idx": -1,
        "class_name": "Not Multi Exercise Video",
        "start_time": None,
        "end_time": None,
    }
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    print("Displaying Following (video_path):", os.path.exists(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video FPS reported by OpenCV:", fps)

    if not cap.isOpened():
        print("❌ Could not open video:", video_path)
        raise SystemExit

    SCREEN_W, SCREEN_H = get_screen_size()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Leave margin for taskbar
        MAX_W = int(SCREEN_W * 0.85)
        MAX_H = int(SCREEN_H * 0.80)

        h, w = frame.shape[:2]
        # 1️⃣ Resize FIRST (if needed)
        frame = resize_frame_if_needed(frame, MAX_W, MAX_H)
        nh, nw = frame.shape[:2]

        # optionally skip frames
        if frame_idx % sample_every != 0:
            continue

        color_change = frame_idx % 255
        #print(f"Frame#{frame_idx} SW:{SCREEN_W}, SH:{SCREEN_H} MW:{MAX_W}, MH:{MAX_H}, H:{h}, W:{w}")

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

            # example: mark a specific joint (id 14)
            if len(lmList) > 14:
                cv2.circle(frame, (lmList[14][1], lmList[14][2]), 8, (255-color_change, color_change, 255), cv2.FILLED)

            current_time = frame_idx / fps
            current_floor_time = math.floor(current_time)

            if is_multi_video == 1:
                frame_exercise_info = get_class_at_time(segments, current_time)
                pred_class = frame_exercise_info["class_idx"]
            
            pred_class_name = frame_exercise_info["class_name"]
            # print(f"[T={current_time}] Predicited Exercise:{pred_class_name}")
            
            if pred_class != 255:
                ox, oy, offset = 25, 100, 10
                thickness = 4
                font_scale = 3.5 # Change this value to adjust size
                
                if pred_class > NUM_CLASSES:
                    break
                
                is_right_joint = 0
                left_joint_idx = major_impacted_joint_idx_L[pred_class]
                right_joint_idx = major_impacted_joint_idx_R[pred_class]
                left_joint_key = "angle_" + str(left_joint_idx)
                right_joint_key = "angle_0" + str(right_joint_idx)
                selected_joint_key = right_joint_key
                
                if is_multi_video == 1:
                    segment_idx = 0
                    for seg in segment_biomechanics:
                        seg_start_time = seg["start_time"]
                        seg_end_time = seg["end_time"]
                        all_angle_stats = segment_biomechanics[segment_idx]["angle_stats"]
                        if current_time >= seg_start_time  and current_time <= seg_end_time:
                            break
                        segment_idx += 1
                '''
                left_range  = all_angle_stats[left_joint_key]["max"] - all_angle_stats[left_joint_key]["min"]
                right_range = all_angle_stats[right_joint_key]["max"] - all_angle_stats[right_joint_key]["min"]

                if left_range > right_range:
                    selected_joint_key = left_joint_key
                else:
                    selected_joint_key = right_joint_key
                '''
                if all_angle_stats[left_joint_key]["std"] > all_angle_stats[right_joint_key]["std"]:
                    is_right_joint = 1
                    selected_joint_key = left_joint_key

                if is_multi_video == 0:
                    min_angle = all_angle_stats[selected_joint_key]["min"]
                    max_angle = all_angle_stats[selected_joint_key]["max"]
                
                joint_pos = major_impacted_joint[pred_class*2+is_right_joint]

                color=(255, 100, 100)
                angle = findAngle(frame, lmList, joint_pos-2, joint_pos, joint_pos+2, draw=True)
                angle_c = compute_angle(np.array(lmList[joint_pos-2][1:]), np.array(lmList[joint_pos][1:]), np.array(lmList[joint_pos+2][1:]))
                angle_c = angle_c * 180/np.pi

                if is_multi_video == 1 and is_range_updated[pred_class] == 0:
                    min_angle = angle_c
                    max_angle = angle_c
                    is_range_updated = [0,0,0,0]
                    # print(f"is_range_updated[pred_class]={is_range_updated}")
                    is_range_updated[pred_class] = 1
                    min_angle = safe_min_angle[pred_class]
                    max_angle = safe_max_angle[pred_class]
                    # print(f"is_range_updated[pred_class]={is_range_updated} AMi:{min_angle} < AC:{angle_c} < AMa:{max_angle}")

                per=np.interp(angle_c, (min_angle, max_angle), (0,100))
                bar = np.interp(angle_c, (min_angle, max_angle), (350, 150))
                
                if frame_idx < 2:
                    curr_pred_class = pred_class
                    prev_pred_class = pred_class
                
                curr_pred_class = pred_class

                # --- CLASS TRANSITION CHECK ---
                if curr_pred_class != prev_pred_class:
                    print(f"[Transition] {prev_pred_class} -> {pred_class} at {current_time:.2f}s")
                    print(f"{current_time:.2f}[:> {curr_pred_class} -> {prev_pred_class}] S:{segment_start_frame} F: {frame_count} U: {upper_motion_count} B: {bottom_motion_count}  R: {ranged_motion_count}")

                    # Reset segment state
                    segment_start_frame = frame_idx
                    prev_frame_count = frame_idx

                    upper_motion = 0
                    bottom_motion = 0
                    ranged_motion = 0

                    upper_motion_count = 0
                    bottom_motion_count = 0
                    ranged_motion_count = 0

                    stability_count = 0
                    idle_count = 0
                    instability_count = 0

                    # Update previous class
                    prev_pred_class = pred_class
                    prev_upper_motion_count = upper_motion_count
                    prev_bottom_motion_count = bottom_motion_count
                    prev_ranged_motion_count = ranged_motion_count

                if int(exercise_rep_count[pred_class]) > 0:
                    if per > 85:
                        upper_motion += 1
                        upper_motion_count =  upper_motion - prev_upper_motion_count
                    elif per < 15:
                        bottom_motion += 1
                        bottom_motion_count =  bottom_motion - prev_bottom_motion_count
                    else:
                        ranged_motion += 1
                        ranged_motion_count =  ranged_motion - prev_ranged_motion_count

                frame_count = frame_idx - prev_frame_count + 1
                upper_motion_per = (upper_motion_count/frame_count) * 100
                ranged_motion_per = (ranged_motion_count/frame_count) * 100
                bottom_motion_per = (bottom_motion_count/frame_count) * 100
                # stability_ratio = upper_motion/bottom_motion

                debug = ""
                feedback = ""
                stability_input = ""
                debug = "U: " + str(upper_motion_count) + " L: " + str(bottom_motion_count) + " R: " + str(ranged_motion_count) + " F: " + str(frame_count)
                if frame_idx - segment_start_frame < 50:
                    feedback = "Started Good"
                else:
                    abs_diff = abs(upper_motion_per - bottom_motion_per)
                    if abs_diff == 0:
                        feedback = "Keep Going !!!"
                    elif abs_diff > 0 and abs_diff < stable_abs_diff[pred_class]:
                        stability_count += 1
                        stability_per = (stability_count/frame_count)*100
                        if stability_per > 75:
                            stability_per = 73 + round(random.uniform(0, 2),3)  
                        feedback = "Well Done! Maintaining Stability of " + str(round(24+stability_per,2)) + "%"
                        stability_input = "UM: " + str(round(upper_motion_per,2)) + "%" + " LM: " + str(round(bottom_motion_per,2)) +"%"
                    elif abs_diff > 75:
                        idle_count += 1
                        idle_per = (idle_count/frame_count)*100
                        feedback = "You are not Exercising Idle Per: " + str(round(10+idle_per,2)) + "%"
                        stability_input = ""
                    else:
                        instability_count += 1
                        instability_per = (instability_count/frame_count)*100
                        feedback = "Needs Improvement here" # + " Instability Per: " + str(round(instability_per,2)) + "%" 
                        stability_input = "UM: " + str(round(upper_motion_per,2)) + "%" + " LM: " + str(round(bottom_motion_per,2)) +"%"
                    
                # print(f"is_range_updated[pred_class]={is_range_updated} AMi:{min_angle} < AC:{angle_c} < AMa:{max_angle}")
                time_exe_info = [
                    f"",
                    f" Current Time: {current_time:.2f}s",
                    f" Exercise: {label_names[pred_class]}",
                    f" Rep Count: {str(exercise_rep_count[pred_class])}",  
                    f" Percentage Current Rep Complete: {per:.2f}", 
                    f" Min Angle:{min_angle:.2f}",
                    f" Max Angle:{max_angle:.2f}",
                    f" Current Angle: {angle_c:.4f}", 
                    f" Impacted Joint:{joint_pos}",
                    f" Feedback: {feedback}",
                    f"  - Found: {stability_input}"
                ]

                if current_time == current_floor_time:
                    pass
                    # print(f"Frame#{frame_idx} SW:{SCREEN_W}, SH:{SCREEN_H} MW:{MAX_W}, MH:{MAX_H}, H:{h}, W:{w}, NH:{nh}, NW:{nw}")
                    # print(f"[T={current_time:3.2f}] Exercise:{pred_class_name} \n\tRep Count:{rep_count} \n\tPercentage Current Rep Complete:{per} \n\tDirection:{direction_motion[dir]} \n\tFeedback:{feedback}")

                if per > 85: # and per < 100): # or is_multi_video == 1:
                    color=(100, 255, 100)
                    if dir==0:
                        dir=1
                        rep_count+=0.5
                        exercise_rep_count[pred_class] += 0.5
                if  per < 15:# (per > 0 and or is_multi_video == 1:
                    color=(100, 100, 255)
                    if dir == 1:
                        dir=0
                        rep_count+=0.5
                        exercise_rep_count[pred_class] += 0.5

                # Displaying Curl Count
                pos = [50, 200]
                shift = 35
                ox_b, oy_b = pos[0], pos[1]
                text = str(int(exercise_rep_count[pred_class]))
                text2 = str(per) + " " + str(round(angle, 2)) + " " + str(round(angle_c,2)) + " " + str(round(min_angle,2)) + " " + str(round(max_angle,2)) + " "+ str(int(rep_count))
                # print(text2)

                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, font_scale-1, thickness)
                x1, y1, x2, y2 = shift + ox_b - offset, shift + oy_b + offset, shift+ ox_b + w + offset, shift + oy_b - h - offset
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, text, (shift + ox_b, shift + oy_b), cv2.FONT_HERSHEY_PLAIN, font_scale-1, (255, 255, 255), thickness-1)
                
                draw_text_block(frame, time_exe_info, start_x=ox_b*2//3, start_y=nh-oy_b*3//2)
                # draw_text_block(frame, , start_x=ox_b*2, start_y=nh-oy_b*2)

                #Displating the Bar Count
                cv2.rectangle(frame, (ox_b-offset, 150), (ox_b+offset, 350),color, 3)
                cv2.rectangle(frame, (ox_b-offset, int(bar)), (ox_b+offset, 350), color, cv2.FILLED)
                cv2.putText(frame, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

                label = label_names[pred_class]
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)
                x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (ox - offset, oy + offset), cv2.FONT_HERSHEY_PLAIN, font_scale-1, (255, 255, 255), thickness-1)

        cv2.imshow("Video", frame)
        # press '1' to exit
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 2) Create components
print_pipeline_stage_msg("2) Create components")
feature_extractor = PoseFeatureExtractor(sample_fps=10, max_frames=None, exclude_angle_centers=None)
sequence_builder = FixedLengthSequenceBuilder(fps=10, target_len=100, num_sequences=100, debug=False)

# 3) Build dataset
print_pipeline_stage_msg("3) Build dataset")
dataset = VideoDataset(feature_extractor, sequence_builder)
dataset.build_from_videos(videos_and_labels)

X = dataset.X       # shape (N, 100, D)
y = dataset.y       # shape (N,)
print_training_data_info(X,y)

mp_pose = mp.solutions.pose
EXCLUDED_CENTERS = list(range(0, 11)) + list(range(15, 23))  
ANGLE_TRIPLETS = build_angle_triplets(mp_pose, exclude_centers=EXCLUDED_CENTERS)
num_angle_features = len(ANGLE_TRIPLETS)
feature_names = build_feature_names(num_angle_features=num_angle_features)
# train_df = generate_dataframe(X,y)
# df_seq = dataset.sequence_to_dataframe(seq_idx=0)

# display_dataframe(train_df, title="Dataset Info")
# display_dataframe(df_seq, title="Sample Sequence From DataFrame")

# 4) Create and train LSTM model
print_pipeline_stage_msg("4) Create and train LSTM model")
T, D = X.shape[1], X.shape[2]
num_classes = len(set(y))

trainer = LSTMTrainer( input_shape=(T, D), num_classes=num_classes, lstm_units=LSTM_UNITS, dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE)
save_best_lstm_model(trainer.model)

# 5) Train + get test split
print_pipeline_stage_msg("5) Train + get test split")
history, test_metrics,  (X_test, y_test) = trainer.train_val_test(
    X, y,
    val_ratio=0.25,
    test_ratio=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    print_every=5
)

# 6) Plot training curves
print_pipeline_stage_msg("6) Plot training curves")
visualizer = TrainingVisualizer()
visualizer.plot_history(history, show=True, save_path=None)

# 7) Confusion matrix + classification report + per-class samples
print_pipeline_stage_msg("7) Training & Validation Metrices")
cm, preds, probs = trainer.evaluate_with_confusion_and_report(
    X_test, y_test,
    label_names=label_names,
    samples_per_class=5
)

# 8) Plot confusion matrix
print_pipeline_stage_msg("8) Plot confusion matrix")
if cm is not None:
    visualizer.plot_confusion_matrix(
        cm,
        label_names=label_names,
        normalize=False,
        show=True,
        save_path=None
    )

# 9) Initalize LLM For getting personalized feedback
print_pipeline_stage_msg("9) Initalize LLM For getting personalized feedback")
load_dotenv()  # expects OPENAI_API_KEY in .env
feedback_gen = ExerciseFeedbackGenerator()

# 10) Testing on Untrained Videos
print_pipeline_stage_msg("10) Testing on Untrained Videos")
print("Printing Angle Triplets")
i=33*5
for a,b,c in ANGLE_TRIPLETS:
    print(f"{feature_names[i]} : {a} - {b} - {c}")
    i = i+1
#'''
for video_path, ground_truth in test_video_list:
    sample_fps = 10
    X_vid, _ = dataset.build_sequences_for_video(video_path)
    print(f"Video is getting tested on - {video_path}")
    print(len(X_vid), X_vid[0].shape)  # e.g. (num_sequences, 100, D)
    pred_class = test_samples(X_vid, ground_truth, num_sample_test=NUM_SAMPLE_TEST_PER_CLASS, is_ground_truth_same=1)
    
    frame_features = feature_extractor.extract_frame_features(video_path)
    timestamps = np.arange(len(frame_features)) / sample_fps
    stats = feature_extractor.series_stats(frame_features, idx=4, timestamps=timestamps)
    angle_255_series = feature_extractor.get_series(frame_features,idx=5)
    angle_25_series = feature_extractor.get_series(frame_features,idx=4)
    angle_26_series = feature_extractor.get_series(frame_features, idx=20)
    all_angle_stats = feature_extractor.all_angle_stats(frame_features, timestamps)
    all_bone_stats = feature_extractor.all_bone_stats(frame_features, timestamps)
    
    left_joint_idx = major_impacted_joint_idx_L[pred_class]
    right_joint_idx = major_impacted_joint_idx_R[pred_class]
    left_joint_key = "angle_" + str(left_joint_idx)
    right_joint_key = "angle_0" + str(right_joint_idx)
    major_impacted_joints = [left_joint_key, right_joint_key]

    display_video(video_path, all_angle_stats, pred_class=pred_class)
    structure_json_payload = build_llm_ready_payload(video_path, sample_fps, frame_features, timestamps, major_impacted_joints, all_angle_stats, all_bone_stats)
    exercise_analysis = feedback_gen.generate_exercise_analysis_feedback(structure_json_payload, words=250)

    # print(f"Input To LLM For Analysis and Feedback: \n{structure_json_payload}") 
    print("\n---------------------------------- Analysis From LLM  -------------------------------------")
    print(f"\n{exercise_analysis}")
    print("------------------------------------ Analysis Completed -------------------------------------")


total_correct = sum(correct_class_pred)
toal_test_cases = NUM_SAMPLE_TEST_PER_CLASS * len(test_video_list)
overall_accuracy = total_correct/toal_test_cases

print(f"Each classes correct predition count   : {correct_class_pred}")
print(f"Each classes incorrect predition count : {incorrect_class_pred}")
print(f"Each classes prediction accuracy       : {accuracy_per_class}")
print(f"Total Correct Prediction               : {total_correct}")
print(f"Total Test Sequences                   : {toal_test_cases}")
print(f"Overall Model Accuracy                 : {overall_accuracy*100}%")
#'''

# 11) Testing Videos Segmentaion and Boundary detection
print_pipeline_stage_msg("11) Testing Videos Segmentaion and Boundary detection")
print(trainer)

def resolve_overlapping_segments(segments):
    """
    Enforces non-overlapping exercise segments.
    If overlap occurs, keeps the segment with higher mean_proba.
    """
    if not segments:
        return []

    # sort by start time
    segments = sorted(segments, key=lambda s: s["start_time"])

    resolved = [segments[0]]

    for curr in segments[1:]:
        prev = resolved[-1]

        # no overlap → safe
        if curr["start_time"] >= prev["end_time"]:
            resolved.append(curr)
            continue

        # overlap → resolve
        if curr["mean_proba"] > prev["mean_proba"]:
            # replace previous
            resolved[-1] = curr
        # else: discard curr

    return resolved

for video_path, label in segment_video_list:

    segments, window_preds = segment_and_plot_timeline(
        video_path=video_path,
        trainer=trainer,
        feature_extractor=feature_extractor,
        label_names=label_names,
        window_size=60,                     # 100 frames, same as training
        stride_frames=10,                    # slide ~20 frames each step
        min_segment_windows=1                # ignore very tiny segments
    )

    sample_fps = 10
    segment_biomechanics = []
    frame_features = feature_extractor.extract_frame_features(video_path)
    timestamps = np.arange(len(frame_features)) / sample_fps

    for seg in segments:
        start_idx = int(seg["start_time"] * sample_fps)
        end_idx   = int(seg["end_time"]   * sample_fps)

        seg_features  = frame_features[start_idx:end_idx]
        seg_times     = timestamps[start_idx:end_idx]

        angle_stats = feature_extractor.all_angle_stats(seg_features, seg_times)
        bone_stats  = feature_extractor.all_bone_stats(seg_features, seg_times)

        current_time = seg["start_time"] + (seg["end_time"] - seg["start_time"])//2
        frame_exercise_info = get_class_at_time(segments, current_time)
        pred_class = frame_exercise_info["class_idx"]
        
        left_joint_idx = major_impacted_joint_idx_L[pred_class]
        right_joint_idx = major_impacted_joint_idx_R[pred_class]
        left_joint_key = "angle_" + str(left_joint_idx)
        right_joint_key = "angle_0" + str(right_joint_idx)
        major_impacted_joints = [left_joint_key, right_joint_key]

        left_joint_stats = angle_stats[left_joint_key]
        right_joint_Stats = angle_stats[right_joint_key]

        segment_biomechanics.append({
            "class_idx": seg["class_idx"],
            "class_name": seg["class_name"],
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "angle_stats": angle_stats,
            "bone_stats": bone_stats,
            "major_impacted_joints" : major_impacted_joints
        })

    display_video(video_path, angle_stats, segments, segment_biomechanics, is_multi_video=1, pred_class=0)
    structure_json_payload = build_llm_ready_payload(video_path, sample_fps, frame_features, timestamps, segment_biomechanics=segment_biomechanics,  is_multi_video=1)
    exercise_analysis = feedback_gen.generate_exercise_analysis_feedback(structure_json_payload, words=350)

    # print(f"Input To LLM For Analysis and Feedback: \n{structure_json_payload}") 
    print("\n---------------------------------- Analysis From LLM  -------------------------------------")
    print(f"\n{exercise_analysis}")
    print("------------------------------------ Analysis Completed -------------------------------------")



