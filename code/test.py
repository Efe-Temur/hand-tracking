import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Hand Tracking
mp_hands_solution = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Gesture classification logic (basic example)
def classify_gesture(landmarks):
    # Convert landmarks to numpy array
    landmarks_np = np.array([(lm.x, lm.y) for lm in landmarks.landmark])
    
    # Calculate relative positions
    wrist = landmarks_np[0]
    index_tip = landmarks_np[8]
    thumb_tip = landmarks_np[4]

    
    
    # Simple rule-based gestures
    if np.linalg.norm(wrist - index_tip) > 0.3:  # Open hand
        return "Open Hand"
    elif np.linalg.norm(thumb_tip - index_tip) < 0.1:  # Fist (thumb close to index tip)
        return "Closed Fist"
    else:
        return "Unknown Gesture"

# Access Webcam
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand landmarks
        result = mp_hands_solution.process(rgb_frame)

        # Draw landmarks and annotate gestures
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Classify the gesture
                gesture = classify_gesture(hand_landmarks)

                # Get the position of the wrist to place the text
                h, w, c = frame.shape
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                # Display the gesture name
                cv2.putText(frame, gesture, (wrist_x - 50, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
