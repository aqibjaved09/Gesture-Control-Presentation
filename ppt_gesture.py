import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import time

# Get screen size for cursor mapping
screen_width, screen_height = pyautogui.size()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Cursor movement smoothing
prev_x, prev_y = 0, 0
smoothing_factor = 8

# Gesture timing
last_gesture_time = 0
gesture_cooldown = 1.2  # seconds

# MediaPipe hand tracking setup
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(max_num_hands=1)
drawer = mp.solutions.drawing_utils

def get_finger_status(hand_landmarks):
    """
    Determines which fingers are up based on landmark positions.
    Returns a list of 0s and 1s (1 for finger up).
    """
    tips = [4, 8, 12, 16, 20]
    finger_states = []

    # Thumb (check horizontal movement)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    # Other fingers (check vertical movement)
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            # Get index finger tip coordinates
            x = hand.landmark[8].x * frame.shape[1]
            y = hand.landmark[8].y * frame.shape[0]

            # Draw finger tip
            cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 255), -1)