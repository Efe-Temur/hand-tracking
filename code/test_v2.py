import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import threading

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

# Gesture classification logic (unchanged)
def classify_gesture(landmarks):
    landmarks_np = np.array([(lm.x, lm.y) for lm in landmarks.landmark])

    wrist = landmarks_np[0]  # Wrist
    index_tip = landmarks_np[8]  # Index Finger Tip
    pinky_tip = landmarks_np[20]  # Pinky Finger Tip
    thumb_tip = landmarks_np[4]  # Thumb Tip
    middle_tip = landmarks_np[12]  # Middle Finger Tip
    ring_tip = landmarks_np[16]  # Ring Finger Tip

    delta_x = index_tip[0] - wrist[0]
    delta_y = index_tip[1] - wrist[1]

    dist_index = np.linalg.norm(wrist - index_tip)
    dist_pinky = np.linalg.norm(wrist - pinky_tip)
    dist_thumb = np.linalg.norm(wrist - thumb_tip)
    dist_middle = np.linalg.norm(wrist - middle_tip)
    dist_ring = np.linalg.norm(wrist - ring_tip)

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
    elif dist_thumb > 0.3 and dist_index > 0.3 and dist_middle > 0.3 and dist_ring > 0.3 and dist_pinky > 0.3:
        return "Open Hand"
    else:
        return "Unknown Gesture"

# UI Functionality
def update_ui(label, gesture_var):
    """
    Update the UI label based on detected gestures.
    """
    while True:
        current_gesture = gesture_var.get()
        label.config(text=f"Gesture: {current_gesture}", font=("Arial", 16))
        label.update()

# Gesture recognition logic (runs in a separate thread)
def gesture_recognition(gesture_var):
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

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    # Classify the gesture
                    gesture = classify_gesture(hand_landmarks)
                    gesture_var.set(gesture)

            cv2.imshow('Hand Gesture Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create the UI
    root = tk.Tk()
    root.title("Gesture-Based Interaction")
    root.geometry("400x200")

    # Initialize gesture variable after root is created
    gesture_var = tk.StringVar(root)
    gesture_var.set("None")

    # Add a label to display the gesture
    label = tk.Label(root, text="Gesture: None", font=("Arial", 16), bg="lightblue")
    label.pack(pady=50)

    # Start gesture recognition in a separate thread
    threading.Thread(target=gesture_recognition, args=(gesture_var,), daemon=True).start()

    # Start the UI update loop
    threading.Thread(target=update_ui, args=(label, gesture_var), daemon=True).start()

    # Run the Tkinter main loop
    root.mainloop()
