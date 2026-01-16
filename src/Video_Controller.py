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

from LSTM import LSTMTrainer
from dataset_manager import VideoDataset
from visualization import TrainingVisualizer
from feature_extractor import PoseFeatureExtractor
from sequence_builder import FixedLengthSequenceBuilder

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

correct_class_pred = [0] * NUM_CLASSES
incorrect_class_pred = [0] * NUM_CLASSES
accuracy_per_class = [0.00] * NUM_CLASSES
total_sample_per_class = [0] * NUM_CLASSES

videos_and_labels = [
    ("data/videos/bicep.mp4", 0),
    ("data/videos/bicep_curl_4.mp4", 0),
    ("data/videos/pushup.mp4", 1),
    ("data/videos/pushup5.mp4", 1),
    ("data/videos/pullup.mp4", 2),
    ("data/videos/pullup5.mp4", 2),
    ("data/videos/squat.mp4", 3),
    ("data/videos/squat2.mp4", 3),
]

test_video_list = [
    ("data/videos/squat5.mp4", 3),
    ("data/videos/pushup7.mp4", 1),
    ("data/videos/bicep_curl5.mp4", 0),
    ("data/videos/bicep_curl_4.mp4", 0),
    ("data/videos/pullup7.mp4", 2),
    ("data/videos/squat3.mp4", 3),
    ("data/videos/pushup3.mp4", 1),
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

def display_video(video_path, pred_class=255):

    frame_idx = 0
    sample_every = 1
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

            # example: mark a specific joint (id 14)
            if len(lmList) > 14:
                cv2.circle(frame, (lmList[14][1], lmList[14][2]), 8, (255-color_change, color_change, 255), cv2.FILLED)

            if pred_class != 255:
                ox, oy, offset = 25, 125, 10
                thickness = 4
                font_scale = 3.5 # Change this value to adjust size
                if pred_class > NUM_CLASSES:
                    break
                label = label_names[pred_class]
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, thickness)
                x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (ox, oy), cv2.FONT_HERSHEY_PLAIN, font_scale-1, (255, 255, 255), thickness-1)

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

# 9) Testing on Untrained Videos
print_pipeline_stage_msg("9) Testing on Untrained Videos")
for video_path, ground_truth in test_video_list:
    X_vid, _ = dataset.build_sequences_for_video(video_path)
    print(f"Video is getting tested on - {video_path}")
    print(len(X_vid), X_vid[0].shape)  # e.g. (num_sequences, 100, D)
    pred_class = test_samples(X_vid, ground_truth, num_sample_test=NUM_SAMPLE_TEST_PER_CLASS, is_ground_truth_same=1)
    display_video(video_path, pred_class)

total_correct = sum(correct_class_pred)
toal_test_cases = NUM_SAMPLE_TEST_PER_CLASS * len(test_video_list)
overall_accuracy = total_correct/toal_test_cases

print(f"Each classes correct predition count   : {correct_class_pred}")
print(f"Each classes incorrect predition count : {incorrect_class_pred}")
print(f"Each classes prediction accuracy       : {accuracy_per_class}")
print(f"Total Correct Prediction               : {total_correct}")
print(f"Total Test Sequences                   : {toal_test_cases}")
print(f"Overall Model Accuracy                 : {overall_accuracy*100}%")