import math
import time

import cv2
import mediapipe as mp
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from constants import *

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize Pycaw for volume control.
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

volume.SetMasterVolumeLevelScalar(INITIAL_VOLUME / 100, None)

new_vol = None

isVolumeControllerOn = False

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

        if handLabel == 'Right':
            # Extract coordinates of the index and thumb fingertips.
            indexFingerTip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumbTip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Calculate the euclidean distance between the index finger and the thumb.
            dx = indexFingerTip.x - thumbTip.x
            dy = indexFingerTip.y - thumbTip.y

            distanceBetweenIndexAndThumb = math.hypot(dx, dy)

            if distanceBetweenIndexAndThumb <= DISTANCE_OFFSET:
                # Toggle the boolean variable on/off
                isVolumeControllerOn = not isVolumeControllerOn

                time.sleep(1)

            if isVolumeControllerOn:
                # Extract coordinates of the middle fingertip and MCP joint.
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

                # Calculate the angle between the middle finger line and the y-axis.
                dx = middle_finger_mcp.x - middle_finger_tip.x
                dy = middle_finger_mcp.y - middle_finger_tip.y

                angle_rad = math.atan2(dx, dy)
                angle_deg = math.degrees(angle_rad)

                # Invert the angles, to accommodate for the inversion of the axes in openCV
                angle_deg *= -1

                # Limit the angle's range to [-90, 90]
                if -90 <= angle_deg <= 90:
                    # Map angle to volume range.
                    new_vol = (angle_deg + 90) / 180 * MAX_VOLUME

                    # Update the system volume based on the calculated angle.
                    volume.SetMasterVolumeLevelScalar(new_vol / 100, None)

                if TESTING:
                    current_vol = round(volume.GetMasterVolumeLevelScalar() * 100)

                    # Put the text on the image
                    cv2.putText(image, f"Volume: {str(current_vol)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if TESTING:
        # Display the image.
        cv2.imshow('Hand Gesture Volume Control', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
