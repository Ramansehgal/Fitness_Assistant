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

    print(f"\n=== Frame #{frame_idx} ===")

    # BGR -> RGB for mediapipe
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    lmList = []
    if results.pose_landmarks:
        # draw skeleton
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        h, w, c = frame.shape
        # print keypoints
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])

            print(f"  kp {id:2d}: norm=({lm.x:.3f}, {lm.y:.3f}, {lm.z:.3f}), "
                  f"pix=({cx:4d}, {cy:4d}), visibility={lm.visibility:.3f}")

        color_change = frame_idx % 255
        # example: mark a specific joint (id 14)
        if len(lmList) > 14:
            cv2.circle(frame, (lmList[14][1], lmList[14][2]), 8, (255-color_change, color_change, 255), cv2.FILLED)

    cv2.imshow("Video", frame)
    # press '1' to exit
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()
