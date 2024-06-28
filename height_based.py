import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

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

# Variables to store the y-coordinate of half of the screen height.
initial_y = None

new_vol = None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    if TESTING:
        # Flip the image horizontally for a later selfie-view display.
        image = cv2.flip(image, 1)

    if initial_y is None:
        initial_y = image.shape[0] // 2
        print('initial: ', initial_y)

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
            # Extract coordinates of the index fingertip.
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Use the y-coordinate of the index fingertip as a reference.
            current_y = index_finger_tip.y * image.shape[0]  # Scale to image height

            # Calculate the difference in y-coordinate.
            y_diff = current_y - initial_y

            min_division_offset = initial_y / INITIAL_VOLUME
            max_division_offset = initial_y / (MAX_VOLUME - INITIAL_VOLUME)

            if current_y > initial_y:
                # Decrease Volume
                new_vol = max(INITIAL_VOLUME - round(abs(y_diff)/min_division_offset), 0)

            elif current_y < initial_y:
                # Increase Volume
                new_vol = min(INITIAL_VOLUME + round(abs(y_diff)/max_division_offset), MAX_VOLUME)

            # Update the system volume based on the direction of finger movement.
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
