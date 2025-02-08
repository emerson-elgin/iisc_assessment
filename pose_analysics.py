import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Define the parent-child joint relationships in MediaPipe
PARENT_CHILD_PAIRS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
]

# Function to compute orientation vector
def compute_orientation(p1, p2):
    return np.array([p2[0] - p1[0], p2[1] - p1[1]])

# Function to compute angle between two vectors
def compute_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-6)  # Avoid division by zero
    return np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi

# Function to draw pose on frame
def draw_pose_on_frame(frame, pose_landmarks):
    if not pose_landmarks:
        return frame  # If no landmarks, return original frame
    
    annotated_frame = frame.copy()
    mp_drawing.draw_landmarks(
        annotated_frame,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
    )
    
    return annotated_frame

# Process video, extract keypoints, and save skeleton video
def process_video(video_path, output_path="output_skeleton.mp4"):
    cap = cv2.VideoCapture(video_path)
    keypoints_history = []
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        keypoints = {}

        if result.pose_landmarks:
            for landmark in mp_pose.PoseLandmark:
                kp = result.pose_landmarks.landmark[landmark]
                keypoints[landmark] = (kp.x, kp.y)
        
        keypoints_history.append(keypoints)
        
        # Draw pose skeleton on frame
        annotated_frame = draw_pose_on_frame(frame, result.pose_landmarks)
        out.write(annotated_frame)

        cv2.imshow("Pose Skeleton", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Skeleton video saved as {output_path}")

    return keypoints_history

# Compute features from keypoints
def compute_features(keypoints_history):
    HOAD2D, HORJO2D, HOJO2D, HOJD2D = [], [], [], []

    for f in range(len(keypoints_history)):
        if f < 10:
            continue  # Skip first 10 frames for reliable angular displacement

        keypoints = keypoints_history[f]
        keypoints_prev = keypoints_history[f - 10]

        angles, displacements, orientations, relative_angles = [], [], [], []

        for parent, child in PARENT_CHILD_PAIRS:
            if (parent in keypoints and child in keypoints and 
                parent in keypoints_prev and child in keypoints_prev):

                v_curr = compute_orientation(keypoints[parent], keypoints[child])
                v_prev = compute_orientation(keypoints_prev[parent], keypoints_prev[child])
                angles.append(compute_angle(v_curr, v_prev))

                displacement = np.linalg.norm(np.array(keypoints[child]) - np.array(keypoints_prev[child]))
                displacements.append(displacement)

                orientation_angle = np.arctan2(v_curr[1], v_curr[0]) * 180 / np.pi
                orientations.append(orientation_angle)

        # Compute pairwise joint orientation angles
        landmark_keys = list(keypoints.keys())
        for i in range(len(landmark_keys)):
            for j in range(i + 1, len(landmark_keys)):
                v1 = compute_orientation(keypoints[landmark_keys[i]], keypoints[landmark_keys[j]])
                angle = np.arctan2(v1[1], v1[0]) * 180 / np.pi
                relative_angles.append(angle)

        HOAD2D.extend(angles)
        HOJD2D.extend(displacements)
        HOJO2D.extend(orientations)
        HORJO2D.extend(relative_angles)

    return HOAD2D, HORJO2D, HOJO2D, HOJD2D

# Plot histograms
def plot_histograms(HOAD2D, HORJO2D, HOJO2D, HOJD2D):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(HOAD2D, bins=30, color='r', alpha=0.7)
    axes[0, 0].set_title('HOAD2D: Histogram of Angular Displacement')

    axes[0, 1].hist(HORJO2D, bins=30, color='b', alpha=0.7)
    axes[0, 1].set_title('HORJO2D: Histogram of Relative Joint Orientation')

    axes[1, 0].hist(HOJO2D, bins=30, color='g', alpha=0.7)
    axes[1, 0].set_title('HOJO2D: Histogram of Joint Orientation')

    axes[1, 1].hist(HOJD2D, bins=30, color='m', alpha=0.7)
    axes[1, 1].set_title('HOJD2D: Histogram of Joint Displacement')

    plt.tight_layout()
    plt.show()

# Run the pipeline
video_path = "baby_movement_video.mp4"  # Update with your actual video path
output_video_path = "skeleton_output.mp4"

keypoints_history = process_video(video_path, output_video_path)
HOAD2D, HORJO2D, HOJO2D, HOJD2D = compute_features(keypoints_history)
plot_histograms(HOAD2D, HORJO2D, HOJO2D, HOJD2D)
