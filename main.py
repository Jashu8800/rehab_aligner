import cv2
import mediapipe as mp
from flask import Flask, request, jsonify, render_template
from threading import Thread
from filterpy.kalman import KalmanFilter
import numpy as np
import math
from datetime import datetime
import pygame  
# Pygame for sound alarm

from flask_cors import CORS

app = Flask(_name_)
CORS(app)

# Initialize MediaPipe Hands and Pose solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Global variables
running = False
feedback_message = ""
posture_message = "Posture: Not Detected"
finger_states = {"Index": False, "Thumb": False, "Pinky": False}
gesture_messages = {
    "Index": "Index Finger Up: Need Water",
    "Thumb": "Thumb Up: Need Food",
    "Pinky": "Pinky Up: Want to go washroom"
}

# Kalman Filter for each hand landmark
kalman_filters = [KalmanFilter(dim_x=4, dim_z=2) for _ in range(21)]

def initialize_kalman_filters():
    for kf in kalman_filters:
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])  # State transition matrix
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])  # Measurement function
        kf.P *= 1000  # Initial covariance matrix
        kf.R = np.eye(2) * 5  # Measurement noise covariance
        kf.Q = np.eye(4) * 0.01  # Process noise covariance

initialize_kalman_filters()

def apply_kalman_filter(landmark, index):
    kf = kalman_filters[index]
    if not hasattr(kf, "x") or kf.x is None:
        kf.x = np.array([landmark.x, landmark.y, 0, 0])
    z = np.array([landmark.x, landmark.y])  # Measurement
    kf.predict()
    kf.update(z)
    return kf.x[:2]  # Smoothed x, y

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    cb = c - b
    cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

detected_parts = {}

def detect_posture(frame):
    global detected_parts, posture_message
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    posture_message = "Posture: Not Detected"
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        posture_message = ""
        
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

        # Check and store available landmarks
        for part in mp_pose.PoseLandmark:
            if landmarks[part].visibility > 0.5:  # Landmark is visible
                detected_parts[part.name] = (landmarks[part].x, landmarks[part].y)

        # Analyze posture based on detected landmarks
        if "LEFT_SHOULDER" in detected_parts and "RIGHT_SHOULDER" in detected_parts:
            left_shoulder = detected_parts["LEFT_SHOULDER"]
            right_shoulder = detected_parts["RIGHT_SHOULDER"]
            if abs(left_shoulder[1] - right_shoulder[1]) < 0.05:
                posture_message += "Shoulders Aligned. "
            else:
                posture_message += "Shoulders Misaligned. "

        if "LEFT_HIP" in detected_parts and "RIGHT_HIP" in detected_parts:
            left_hip = detected_parts["LEFT_HIP"]
            right_hip = detected_parts["RIGHT_HIP"]
            if abs(left_hip[1] - right_hip[1]) < 0.05:
                posture_message += "Hips Aligned. "
            else:
                posture_message += "Hips Misaligned. "

        if (
            "LEFT_HIP" in detected_parts
            and "LEFT_KNEE" in detected_parts
            and "LEFT_ANKLE" in detected_parts
        ):
            left_hip = detected_parts["LEFT_HIP"]
            left_knee = detected_parts["LEFT_KNEE"]
            left_ankle = detected_parts["LEFT_ANKLE"]
            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            if 85 <= knee_angle <= 95:
                posture_message += "Left Knee at 90°. "
            else:
                posture_message += "Left Knee Misaligned. "

        if not posture_message:
            posture_message = "Good Posture!"
        else:
            posture_message = "Posture Issues: " + posture_message


def display_message(frame, text, position=(50, 50), color=(0, 0, 255), size=0.5):
    """Utility function to display messages on the frame."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

def is_finger_up(tip, dip, threshold=0.02):
    return (tip[1] - dip[1]) < -threshold

# Logs to store patient feedback and posture information
logss = []

def save_log(feedback, posture):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feedback": feedback,
        "posture": posture
    }

    # Check if the log is unique before appending
    if not any(log["timestamp"] == log_entry["timestamp"] and log["feedback"] == log_entry["feedback"] for log in logss):
        logss.append(log_entry)
        with open("logss.txt", "a") as log_file:
            log_file.write(f"{log_entry['timestamp']}: {log_entry['feedback']} - {log_entry['posture']}\n")

# Fall detection (when the patient falls)
import pyttsx3
import time
from threading import Thread

# Global variables
falling_state = False  # To track if the patient is falling
alarm_active = False  # To track if the alarm is already active

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speaking rate
tts_engine.setProperty('volume', 1.0)  # Set volume to maximum

def sound_alarm():
    """Alert the user repeatedly when the patient is falling."""
    global alarm_active, falling_state

    # Function to handle the sound
    def play_alarm():
        while True:
            if falling_state:  # Keep playing the alarm while the patient is falling
                tts_engine.say("Patient falling! Please assist.")
                tts_engine.runAndWait()
                time.sleep(1)  # Repeat every second
            time.sleep(0.1)  # Check the falling state every 100ms

    if not alarm_active:  # Start the alarm only if it's not already active
        alarm_active = True
        alarm_thread = Thread(target=play_alarm)
        alarm_thread.daemon = True  # Ensure thread exits when program terminates
        alarm_thread.start()

# Fall detection function
def detect_fall(detected_parts):
    global falling_state

    if "LEFT_SHOULDER" in detected_parts and "RIGHT_SHOULDER" in detected_parts:
        left_shoulder = detected_parts["LEFT_SHOULDER"]
        right_shoulder = detected_parts["RIGHT_SHOULDER"]

        # Check if shoulders are significantly misaligned (indicative of a fall)
        if abs(left_shoulder[1] - right_shoulder[1]) > 0.2:  # Threshold for fall
            if not falling_state:  # If fall just started
                falling_state = True
                sound_alarm()  # Trigger the alarm
            return True

    # If shoulders are aligned again, stop the fall detection and reset the state
    falling_state = False
    return False

def detect_hand_and_posture():
    global running, feedback_message, posture_message
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand detection
        hand_results = hands.process(rgb_frame)
        feedback_message = ""
        detect_posture(frame)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                smoothed_landmarks = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    smoothed_x, smoothed_y = apply_kalman_filter(landmark, idx)
                    smoothed_landmarks.append((smoothed_x, smoothed_y))

                index_tip = smoothed_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_dip = smoothed_landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                thumb_tip = smoothed_landmarks[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = smoothed_landmarks[mp_hands.HandLandmark.THUMB_IP]
                pinky_tip = smoothed_landmarks[mp_hands.HandLandmark.PINKY_TIP]
                pinky_dip = smoothed_landmarks[mp_hands.HandLandmark.PINKY_DIP]

                finger_states["Index"] = is_finger_up(index_tip, index_dip)
                finger_states["Thumb"] = is_finger_up(thumb_tip, thumb_ip, threshold=0.03)
                finger_states["Pinky"] = is_finger_up(pinky_tip, pinky_dip)

                for finger, state in finger_states.items():
                    if state:
                        feedback_message = gesture_messages[finger]
                        break

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

        # Fall detection
        if detect_fall(detected_parts):
            feedback_message = "Patient Falling!"
            
        # Save log
        save_log(feedback_message, posture_message)

        # Display feedback and posture messages
        display_message(frame, feedback_message, position=(50, 100), color=(0, 0, 255))
        display_message(frame, posture_message, position=(50, 150), color=(0, 0, 255))
        cv2.namedWindow('Rehab Aligner - Hand and Posture Detection', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Rehab Aligner - Hand and Posture Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow('Rehab Aligner - Hand and Posture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_rehab_aligner():
    global running
    if not running:
        running = True
        Thread(target=detect_hand_and_posture).start()
        return jsonify({"status": "Started Rehab Aligner"})
    return jsonify({"status": "Already Running"})

@app.route('/stop', methods=['POST'])
def stop_rehab_aligner():
    global running
    running = False
    return jsonify({"status": "Stopped Rehab Aligner"})

@app.route('/get_logs', methods=['GET'])
def get_logs():
    return jsonify({"logss": logss})

if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True)