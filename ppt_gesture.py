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
            # Get current finger states
            fingers = get_finger_status(hand)

            # Cursor movement (only index finger up)
            if fingers == [0, 1, 0, 0, 0]:
                mapped_x = np.interp(x, (0, frame.shape[1]), (0, screen_width))
                mapped_y = np.interp(y, (0, frame.shape[0]), (0, screen_height))
                smooth_x = prev_x + (mapped_x - prev_x) / smoothing_factor
                smooth_y = prev_y + (mapped_y - prev_y) / smoothing_factor

                pyautogui.moveTo(smooth_x, smooth_y)
                prev_x, prev_y = smooth_x, smooth_y
               
                # Detect gestures and perform actions
            current_time = time.time()

            # 1. Start Presentation (5 fingers)
            if fingers.count(1) == 5 and current_time - last_gesture_time > gesture_cooldown:
                pyautogui.press('f5')
                last_gesture_time = current_time
                cv2.putText(frame, "Start Presentation", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
               
                # 2. Next Slide (index up only)
            elif fingers[1] == 1 and fingers[2] == 0 and current_time - last_gesture_time > gesture_cooldown:
                pyautogui.press('right')
                last_gesture_time = current_time
                cv2.putText(frame, "Next Slide", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # 3. Previous Slide (index and middle fingers up)
            elif fingers[1] == 1 and fingers[2] == 1 and current_time - last_gesture_time > gesture_cooldown:
                pyautogui.press('left')
                last_gesture_time = current_time
                cv2.putText(frame, "Previous Slide", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                  # 4. Exit using 4 fingers (no thumb)
            elif fingers == [0, 1, 1, 1, 1] and current_time - last_gesture_time > gesture_cooldown:
                pyautogui.press('esc')
                last_gesture_time = current_time
                cv2.putText(frame, "Exit (4 Fingers)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                 # 5. Swipe Exit (based on wrist movement)
            wrist_x = hand.landmark[0].x * frame.shape[1]
            swipe_distance = wrist_x - prev_x
            if abs(swipe_distance) > 150 and current_time - last_gesture_time > gesture_cooldown:
                pyautogui.press('esc')
                last_gesture_time = current_time
                cv2.putText(frame, "Exit by Swipe", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)



                drawer.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Controlled Presentation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# for clean up
cap.release()
cv2.destroyAllWindows()