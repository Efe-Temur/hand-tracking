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

# Gesture classification logic
def classify_gesture(landmarks):
    # Convert landmarks to numpy array for easy access
    landmarks_np = np.array([(lm.x, lm.y) for lm in landmarks.landmark])

    # Key landmarks for gesture recognition
    wrist = landmarks_np[0]  # Wrist
    index_tip = landmarks_np[8]  # Index Finger Tip
    pinky_tip = landmarks_np[20]  # Pinky Finger Tip
    thumb_tip = landmarks_np[4]  # Thumb Tip
    middle_tip = landmarks_np[12]  # Middle Finger Tip
    ring_tip = landmarks_np[16]  # Ring Finger Tip

    # Calculate relative positions of index tip to wrist
    delta_x = index_tip[0] - wrist[0]
    delta_y = index_tip[1] - wrist[1]

    # Calculate distances based on normalized coordinates (not absolute pixel distances)
    dist_index = np.linalg.norm(wrist - index_tip)
    dist_pinky = np.linalg.norm(wrist - pinky_tip)
    dist_thumb = np.linalg.norm(wrist - thumb_tip)
    dist_middle = np.linalg.norm(wrist - middle_tip)
    dist_ring = np.linalg.norm(wrist - ring_tip)

    
    
    # Calculate vectors
    # Vector from wrist to index tip
    vector_index = np.array([index_tip[0] - wrist[0], index_tip[1] - wrist[1]])

    # Angle of the index finger relative to the x-axis (for left/right)
    angle_x = np.arctan2(vector_index[1], vector_index[0])  # atan2 gives angle in radians
    # Angle of the index finger relative to the y-axis (for up/down)
    angle_y = np.arctan2(vector_index[0], vector_index[1])

    # Convert angles to degrees
    angle_x_deg = np.degrees(angle_x)
    angle_y_deg = np.degrees(angle_y)
    
    # Gesture definitions based on relative distances between landmarks
    if dist_index > 0.25 and dist_pinky < 0.15 and dist_thumb < 0.15:
        if abs(delta_x) < 0.2 and delta_y > 0.2:
            return "Pointing Down"
        elif abs(delta_x) < 0.2 and delta_y < -0.2:
            return "Pointing Up"
        elif delta_x > 0.2 and abs(delta_y) < 0.2:
            return "Pointing Right"
        elif delta_x < -0.2 and abs(delta_y) < 0.2:
            return "Pointing Left"
        
    elif dist_pinky > 0.25 and dist_index < 0.15 and dist_thumb < 0.15:
        return "Pointing Pinky Finger"
    elif dist_index > 0.25 and dist_middle > 0.25 and dist_ring < 0.2 and dist_pinky < 0.2:
        return "Victory Pose"
    elif dist_thumb > 0.25 and dist_index < 0.2 and dist_pinky < 0.2:
        return "Thumbs Up"
    elif dist_thumb > 0.25 and dist_pinky > 0.25 and dist_index < 0.2 and dist_middle < 0.2:
        return "Aloha Sign"
    elif dist_thumb < 0.15 and dist_index < 0.15 and dist_middle < 0.15 and dist_ring < 0.15 and dist_pinky < 0.15:
        return "Closed Fist"
    elif dist_thumb > 0.2 and dist_index > 0.2 and dist_middle > 0.2 and dist_ring > 0.2 and dist_pinky > 0.2:
        return "Open Hand"
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
