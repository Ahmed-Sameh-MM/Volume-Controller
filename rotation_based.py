import math
import time

import cv2
import mediapipe as mp

# Project Imports (My own files)
from constants import *
from mode import Mode
from audio import Audio
from mouse import Mouse

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

audio = Audio()

audio.set_volume(value=INITIAL_VOLUME / 100)

new_vol = None

mode: Mode = None

previous_y = None

mouse = Mouse()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    if TESTING:
        # Flip the image horizontally for a later selfie-view display.
        image = cv2.flip(image, 1)

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands.
    results = hands.process(image)

    if TESTING:
        # Convert the RGB image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        if TESTING:
            # Draw hand landmarks on the image.
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        handLabel = results.multi_handedness[0].classification[0].label

        if TESTING and handLabel == 'Right' or (not TESTING) and handLabel == 'Left':
            # Extract coordinates of all fingertips.
            thumbTip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            indexFingerTip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middleFingerTip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ringFingerTip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinkyTip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Calculate the euclidean distance between the index finger and the thumb. (Mode 1)
            dx = indexFingerTip.x - thumbTip.x
            dy = indexFingerTip.y - thumbTip.y
            distanceBetweenIndexAndThumb = math.hypot(dx, dy)

            # Calculate the euclidean distance between the ring finger and the thumb. (Mode 2)
            dx = ringFingerTip.x - thumbTip.x
            dy = ringFingerTip.y - thumbTip.y
            distanceBetweenRingAndThumb = math.hypot(dx, dy)

            # Calculate the euclidean distance between the middle finger and the thumb.
            dx = middleFingerTip.x - thumbTip.x
            dy = middleFingerTip.y - thumbTip.y
            distanceBetweenMiddleAndThumb = math.hypot(dx, dy)

            if distanceBetweenIndexAndThumb <= DISTANCE_OFFSET:
                # Toggle the mode variable
                if mode is None:
                    mode = Mode.VOLUME_MODE
                    time.sleep(1)

                elif mode == Mode.VOLUME_MODE:
                    mode = None
                    time.sleep(1)

            elif distanceBetweenRingAndThumb <= DISTANCE_OFFSET:
                # Toggle the mode variable
                if mode is None:
                    mode = Mode.MOUSE_NAVIGATION_MODE
                    time.sleep(1)

                elif mode == Mode.MOUSE_NAVIGATION_MODE:
                    mode = None
                    time.sleep(1)

            if mode == Mode.VOLUME_MODE:
                # Extract coordinates of the middle fingertip and MCP joint.
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                # Calculate the angle between the middle finger line and the y-axis.
                dx = middle_finger_mcp.x - middle_finger_tip.x
                dy = middle_finger_mcp.y - middle_finger_tip.y

                angle_rad = math.atan2(dx, dy)
                angle_deg = math.degrees(angle_rad)

                if TESTING:
                    # Invert the angles, to accommodate for the horizontal flip of the image
                    angle_deg *= -1

                # Bound the angle's range from -ve rotation offset to +ve rotation offset
                if -ROTATION_ANGLE_OFFSET <= angle_deg <= ROTATION_ANGLE_OFFSET:
                    # Map angle to volume range.
                    new_vol = (angle_deg + ROTATION_ANGLE_OFFSET) / (ROTATION_ANGLE_OFFSET * 2) * MAX_VOLUME

                    # Update the system volume based on the calculated angle.
                    audio.set_volume(value=new_vol / 100)

                if TESTING:
                    current_vol = audio.get_volume()

                    # Put the text on the image
                    cv2.putText(image, f"Volume: {str(current_vol)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            elif mode == Mode.MOUSE_NAVIGATION_MODE:
                if distanceBetweenIndexAndThumb <= DISTANCE_OFFSET:
                    if previous_y is None:
                        previous_y = indexFingerTip.y * image.shape[0]

                    # Use the y-coordinate of the index fingertip as a reference, as the two fingertips are very close.
                    current_y = indexFingerTip.y * image.shape[0]  # Scale to image height

                    y_diff = current_y - previous_y

                    mouse.scroll(distance=y_diff)

                    mouse.move(finger_tip=indexFingerTip)

                    previous_y = current_y

                elif distanceBetweenMiddleAndThumb <= DISTANCE_OFFSET:
                    mouse.click()

                else:
                    mouse.move(finger_tip=indexFingerTip)

                    if previous_y is not None:
                        previous_y = None

    if TESTING:
        # Display the image.
        cv2.imshow('Hand Gesture Volume Control', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
