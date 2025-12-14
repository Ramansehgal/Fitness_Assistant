# video_segmentation.py

import numpy as np
import matplotlib.pyplot as plt

def resample_to_length(feats, target_len):
    """
    feats: (T, D) per-frame features
    target_len: desired number of frames

    Returns: (target_len, D)
    - If T > target_len: uniformly downsample.
    - If T < target_len: uniformly sample with repetition.
    """
    feats = np.asarray(feats, dtype=np.float32)
    T, D = feats.shape
    if T == 0:
        return np.zeros((target_len, D), dtype=np.float32)
    if T == target_len:
        return feats.copy()

    idxs = np.linspace(0, T - 1, target_len)
    idxs = np.round(idxs).astype(int)
    idxs = np.clip(idxs, 0, T - 1)
    return feats[idxs]

def build_per_second_labels(window_preds, sample_fps, video_duration_s):
    """
    window_preds: list of dicts with start_time, end_time, class_idx, proba
    Returns: per_second_labels, per_second_confidence
    """
    sec_votes = {sec: [] for sec in range(int(video_duration_s) + 1)}

    for w in window_preds:
        start_sec = int(np.floor(w["start_time"]))
        end_sec = int(np.ceil(w["end_time"]))
        for sec in range(start_sec, end_sec):
            sec_votes[sec].append((w["class_idx"], w["proba"]))

    per_sec_label = {}
    per_sec_conf = {}

    for sec, votes in sec_votes.items():
        if not votes:
            continue
        classes, probs = zip(*votes)
        # weighted majority vote
        uniq = set(classes)
        scores = {
            c: sum(p for (cl, p) in votes if cl == c)
            for c in uniq
        }
        best_class = max(scores, key=scores.get)
        per_sec_label[sec] = best_class
        per_sec_conf[sec] = scores[best_class] / len(votes)

    return per_sec_label, per_sec_conf

def collapse_time_segments(per_sec_label, label_names, min_duration=2):
    segments = []
    secs = sorted(per_sec_label.keys())
    if not secs:
        return segments

    cur_class = per_sec_label[secs[0]]
    start_sec = secs[0]

    for s in secs[1:]:
        if per_sec_label[s] != cur_class:
            end_sec = s
            if end_sec - start_sec >= min_duration:
                segments.append({
                    "start_time": start_sec,
                    "end_time": end_sec,
                    "class_idx": cur_class,
                    "class_name": label_names[cur_class]
                })
            cur_class = per_sec_label[s]
            start_sec = s

    segments.append({
        "start_time": start_sec,
        "end_time": secs[-1] + 1,
        "class_idx": cur_class,
        "class_name": label_names[cur_class]
    })

    return segments


def segment_video_exercises(
    video_path,
    trainer,
    feature_extractor,
    label_names=None,
    window_size=None,      # in frames (features), default: trainer.input_shape[0]
    stride_frames=None,    # in frames, default: window_size // 4
    min_segment_windows=1  # minimum number of windows to keep a segment
):
    """
    High-level:
      1. Extract frame_features for the entire video
      2. Slide a window over time
      3. Predict class for each window
      4. Run-length encode to find segments with same class

    Returns:
      segments: list of dicts, each like:
          {
            "class_idx": int,
            "class_name": str or None,
            "start_time": float (sec),
            "end_time": float (sec),
            "mean_proba": float,
            "start_window": int,
            "end_window": int
          }
      window_preds: list of (time_center_sec, class_idx, proba, class_name)
    """

    # 1) per-frame features for full video
    frame_features = feature_extractor.extract_frame_features(video_path)
    if frame_features.size == 0:
        print(f"[segment_video_exercises] No frame features for {video_path}.")
        return [], []

    T, D = frame_features.shape
    sample_fps = feature_extractor.sample_fps  # frames per second in feature space
    video_duration_s = T / sample_fps + (T%sample_fps)/sample_fps

    print(f"[segment_video_exercises] {video_path}: T={T}, D={D}, sample_fps={sample_fps} video duration  ={video_duration_s}s")

    # 2) window parameters
    if window_size is None:
        # use same time length as training
        window_size = trainer.input_shape[0]   # e.g., 100 frames
    if stride_frames is None:
        stride_frames = max(1, window_size // 4)  # 75% overlap by default

    if T < 2:
        print("[segment_video_exercises] Too few frames for segmentation.")
        return [], []

    window_preds = []  # (center_time_sec, class_idx, proba, class_name)

    # 3) slide window
    starts = list(range(0, max(1, T - window_size + 1), stride_frames))
    if len(starts) == 0:
        starts = [0]

    for start in starts:
        end = start + window_size
        if end <= T:
            window = frame_features[start:end]  # (window_size, D)
        else:
            # tail: use remaining frames and resample to window_size
            window = frame_features[start:T]
            window = resample_to_length(window, window_size)

        start_time = start / sample_fps
        end_time = min(end, T) / sample_fps

        pred_idx, pred_proba, pred_name = trainer.predict_sequence(
            window, label_names=label_names
        )

        window_preds.append({
            "start_time": start_time,
            "end_time": end_time,
            "class_idx": pred_idx,
            "class_name": pred_name,
            "proba": pred_proba
        })

    print("[DEBUG] Window predictions:")
    for i, w in enumerate(window_preds):
        print(
            f"  win {i:02d} | "
            f"{w['start_time']:5.2f}s → {w['end_time']:5.2f}s | "
            f"class={w['class_idx']} ({w['class_name']}), "
            f"proba={w['proba']:.3f}"
        )

    # 4) run-length encode segments by predicted class
    segments = []
    if not window_preds:
        return segments, window_preds

    current_class = window_preds[0]["class_idx"]
    current_probs = [window_preds[0]["proba"]]
    start_time = window_preds[0]["start_time"]
    end_time = window_preds[0]["end_time"]
    seg_start_window = 0

    for i in range(1, len(window_preds)):
        w_i = window_preds[i]
        class_i = w_i["class_idx"]
        proba_i = w_i["proba"]
        if class_i == current_class:
            current_probs.append(proba_i)
        else:
            # finalize previous segment
            seg_end_window = i - 1
            if (seg_end_window - seg_start_window + 1) >= min_segment_windows:
                # start_time = window_preds[seg_start_window][0]
                # end_time = window_preds[seg_end_window][0]
                end_time = window_preds[seg_end_window]["end_time"]

                mean_proba = float(np.mean(current_probs))
                class_name = (
                    label_names[current_class]
                    if label_names is not None and current_class < len(label_names)
                    else str(current_class)
                )
                segments.append({
                    "class_idx": current_class,
                    "class_name": class_name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "mean_proba": mean_proba,
                    "start_window": seg_start_window,
                    "end_window": seg_end_window,
                })

            # start new segment
            current_class = class_i
            current_probs = [proba_i]
            seg_start_window = i
            start_time = window_preds[seg_start_window]["start_time"]

    # finalize last segment
    seg_end_window = len(window_preds) - 1
    if (seg_end_window - seg_start_window + 1) >= min_segment_windows:
        # start_time = window_preds[seg_start_window][0]
        # end_time = window_preds[seg_end_window][0]
        end_time = window_preds[seg_end_window]["end_time"]
        
        mean_proba = float(np.mean(current_probs))
        class_name = (
            label_names[current_class]
            if label_names is not None and current_class < len(label_names)
            else str(current_class)
        )
        segments.append({
            "class_idx": current_class,
            "class_name": class_name,
            "start_time": start_time,
            "end_time": end_time,
            "mean_proba": mean_proba,
            "start_window": seg_start_window,
            "end_window": seg_end_window,
        })

    # 5) print textual summary
    print(f"\n[segment_video_exercises] Segmentation for {video_path}:")
    for seg in segments:
        print(
            f"  {seg['start_time']:6.2f}s → {seg['end_time']:6.2f}s"
            f"  | class={seg['class_idx']:2d} ({seg['class_name']:15})"
            f"  | mean_proba={seg['mean_proba']:.3f}"
        )

    per_sec_label, per_sec_conf = build_per_second_labels(window_preds, sample_fps, video_duration_s)
    non_overlaping_segments = collapse_time_segments(per_sec_label, label_names, min_duration=2)

    # 6) print non overlapping textual summary
    print(f"\n[segment_video_exercises] Segmentation for {video_path}:")
    for seg in non_overlaping_segments:
        print(
            f"  {seg['start_time']:6.2f}s → {seg['end_time']:6.2f}s"
            f"  | class={seg['class_idx']:2d} ({seg['class_name']:15})"
        )
        
    return segments, window_preds

def segment_and_plot_timeline(
    video_path,
    trainer,
    feature_extractor,
    label_names=None,
    window_size=None,
    stride_frames=None,
    min_segment_windows=1,
    normalize_y=False
):
    """
    Convenience wrapper:
      - runs segment_video_exercises
      - plots predicted class over time as a timeline
    """
    segments, window_preds = segment_video_exercises(
        video_path=video_path,
        trainer=trainer,
        feature_extractor=feature_extractor,
        label_names=label_names,
        window_size=window_size,
        stride_frames=stride_frames,
        min_segment_windows=min_segment_windows
    )

    if not window_preds:
        print("[segment_and_plot_timeline] No window predictions; nothing to plot.")
        return segments, window_preds

    times_start = np.array([w["start_time"] for w in window_preds])
    times_end   = np.array([w["end_time"] for w in window_preds])
    classes     = np.array([w["class_idx"] for w in window_preds], dtype=int)
    probs       = np.array([w["proba"] for w in window_preds])
    times = 0.5 * (times_start + times_end)

    plt.figure(figsize=(10, 4))
    if normalize_y:
        # plot probabilities; class index via color/marker would be a more complex plot
        plt.plot(times, probs, marker="o", linestyle="-")
        plt.ylabel("Confidence")
    else:
        # plot class index as step function
        plt.step(times, classes, where="mid")
        plt.ylabel("Predicted class")

        # nice y-ticks with label names if provided
        if label_names is not None:
            unique_classes = sorted(set(classes.tolist()))
            yticks = unique_classes
            ylabels = [label_names[c] if c < len(label_names) else str(c) for c in unique_classes]
            plt.yticks(yticks, ylabels)

    plt.xlabel("Time (s)")
    plt.title(f"Predicted exercise over time: {video_path}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return segments, window_preds
