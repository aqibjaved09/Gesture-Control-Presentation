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