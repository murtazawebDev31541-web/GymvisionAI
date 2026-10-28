import cv2
import numpy as np
from ultralytics import YOLO


def calculate_angle(a, b, c):
    """Calculates the angle between three points (a, b, c) with b as the vertex."""
    a = np.array(a)  # First point (e.g., Hip)
    b = np.array(b)  # Vertex point (e.g., Knee)
    c = np.array(c)  # End point (e.g., Ankle)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


class ExerciseAnalyzer:

    def __init__(self, exercise_type='squat', fps=30):
        self.exercise_type = exercise_type
        self.counter = 0
        self.stage = 'up'  # Start expecting an 'up' posture
        self.form_status = 'Ready'
        self.is_active = False
        
        # Frame-rate based time tracking
        self.fps = fps if fps > 0 else 30
        self.frame_count = 0
        self.set_duration = 0

    def start_set(self):
        """Starts or resets the workout set timer and counters."""
        self.is_active = True
        self.counter = 0
        self.stage = 'up'
        self.form_status = 'Good Form'
        self.frame_count = 0
        self.set_duration = 0

    def update_duration(self):
        """Calculates the exact set duration based on processed video frames / FPS."""
        if self.is_active:
            self.frame_count += 1
            self.set_duration = int(self.frame_count / self.fps)

    def analyze_pose(self, keypoints):
        """Analyzes COCO keypoints for rep counting and form evaluation.

        COCO Keypoints Index Reference:
        5: Left Shoulder, 6: Right Shoulder
        7: Left Elbow, 8: Right Elbow
        9: Left Wrist, 10: Right Wrist
        11: Left Hip, 12: Right Hip
        13: Left Knee, 14: Right Knee
        15: Left Ankle, 16: Right Ankle
        """
        if keypoints is None or len(keypoints) < 17:
            return

        try:
            if self.exercise_type == 'squat':
                # Extract left leg coordinates
                hip = keypoints[11][:2]
                knee = keypoints[13][:2]
                ankle = keypoints[15][:2]

                angle = calculate_angle(hip, knee, ankle)

                # Strict Full Repetition State Machine Logic:
                # 1. User must go down (angle < 90) while currently 'up'
                if angle < 90 and self.stage == 'up':
                    self.stage = 'down'
                
                # 2. Rep is only counted when user returns completely back 'up' (angle > 160) from 'down'
                elif angle > 160 and self.stage == 'down':
                    self.stage = 'up'
                    self.counter += 1

                # Form Validation Logic
                if angle < 70:
                    self.form_status = 'Too Low / Check Depth'
                elif self.stage == 'down' and angle > 110:
                    self.form_status = 'Go Lower'
                else:
                    self.form_status = 'Correct Form'

            elif self.exercise_type == 'pushup':
                # Extract left arm coordinates
                shoulder = keypoints[5][:2]
                elbow = keypoints[7][:2]
                wrist = keypoints[9][:2]

                angle = calculate_angle(shoulder, elbow, wrist)

                # Strict Full Repetition State Machine Logic for Pushups:
                if angle < 90 and self.stage == 'up':
                    self.stage = 'down'
                elif angle > 160 and self.stage == 'down':
                    self.stage = 'up'
                    self.counter += 1

                if self.stage == 'down' and angle > 100:
                    self.form_status = 'Bend Elbows More'
                else:
                    self.form_status = 'Correct Form'

            # Update accurate set duration per analyzed frame
            self.update_duration()

        except Exception as e:
            print(f'Error analyzing pose keypoints: {e}')


def process_frame(model, frame, analyzer):
    """Runs YOLO11-Pose inference, updates analytics, and overlays metrics onto the frame."""
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()  # YOLO built-in skeleton plotting

    # Extract pose keypoints if detected
    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        if not analyzer.is_active:
            analyzer.start_set()
        analyzer.analyze_pose(keypoints)

    # Overlay metrics on the video frame
    cv2.putText(
        annotated_frame,
        f'Reps: {analyzer.counter}',
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated_frame,
        f'Stage: {analyzer.stage or "Initializing"}',
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated_frame,
        f'Form: {analyzer.form_status}',
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated_frame,
        f'Duration: {analyzer.set_duration}s',
        (30, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated_frame