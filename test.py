import cv2
import mediapipe as mp

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def detect_flat_hand(image):
    # Convert image to RGB format (MediaPipe requires RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands model
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Create a list to store the landmark positions
            landmark_list = []
            for lm in hand_landmarks.landmark:
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append((cx, cy))

            # Check fingers (assuming landmarks are ordered as in MediaPipe docs)
            fingers = {
                "index": [8, 7, 6, 5],
                "middle": [12, 11, 10, 9],
                "ring": [16, 15, 14, 13],
                "pinky": [20, 19, 18, 17],
            }

            for finger_name, landmarks_indices in fingers.items():
                # Extract the landmarks for the current finger
                finger_landmarks = [landmark_list[idx] for idx in landmarks_indices]

                # Check if the finger is flat (all tips are above the other points in the finger)
                flat_finger = True
                for i in range(len(finger_landmarks)):
                    for j in range(i + 1, len(finger_landmarks)):
                        if finger_landmarks[i][1] > finger_landmarks[j][1]:
                            flat_finger = False
                            break
                    if not flat_finger:
                        break

                if flat_finger:
                    print(f"{finger_name.capitalize()} finger is flat!")

            # Draw landmarks on the image (optional)
            for landmark in landmark_list:
                cv2.circle(image, landmark, 5, (0, 255, 0), -1)

    return image


# Capture video from webcam (you can also use cv2.imread() for images)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a selfie view
    frame = cv2.flip(frame, 1)

    # Detect and display the flat hand structure
    processed_frame = detect_flat_hand(frame)

    # Display the processed frame
    cv2.imshow('Flat Hand Detection', processed_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
