import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import keyboard
import time  # Import time module for timer functionality

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

def classify_gesture(landmarks):
    landmarks_np = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    wrist = landmarks_np[0]
    index_tip = landmarks_np[8]
    pinky_tip = landmarks_np[20]
    thumb_tip = landmarks_np[4]
    middle_tip = landmarks_np[12]
    ring_tip = landmarks_np[16]

    # Get Euclidean distance with the Z-axis included
    dist_index = np.linalg.norm(wrist - index_tip)
    dist_pinky = np.linalg.norm(wrist - pinky_tip)
    dist_thumb = np.linalg.norm(wrist - thumb_tip)
    dist_middle = np.linalg.norm(wrist - middle_tip)
    dist_ring = np.linalg.norm(wrist - ring_tip)

    # Adjusted thresholds for distance comparisons
    dist_threshold = 0.2  # Increase the distance threshold to reduce sensitivity

    if dist_index > dist_threshold and dist_pinky < dist_threshold and dist_thumb < dist_threshold and dist_middle < dist_threshold and dist_ring < dist_threshold:
        if abs(index_tip[0] - wrist[0]) < dist_threshold and index_tip[1] - wrist[1] > dist_threshold:
            return "Pointing Down"
        elif abs(index_tip[0] - wrist[0]) < dist_threshold and index_tip[1] - wrist[1] < -dist_threshold:
            return "Pointing Up"
        elif index_tip[0] - wrist[0] > dist_threshold and abs(index_tip[1] - wrist[1]) < dist_threshold:
            return "Pointing Right"
        elif index_tip[0] - wrist[0] < -dist_threshold and abs(index_tip[1] - wrist[1]) < dist_threshold:
            return "Pointing Left"

    elif dist_pinky > 0.2 and dist_index < 0.15 and dist_thumb < 0.15 and dist_middle < 0.15 and dist_ring < 0.15:
        return "Pointing Pinky Finger"

    elif dist_index > 0.25 and dist_middle > 0.25 and dist_ring < 0.2 and dist_pinky < 0.2 and dist_thumb < 0.15:
        return "Victory Pose"

    elif dist_thumb > 0.25 and dist_index < 0.2 and dist_pinky < 0.2:
        return "Thumbs Up"

    elif dist_thumb > 0.2 and dist_pinky > 0.2 and dist_index < 0.2 and dist_middle < 0.2:
        return "Chill"

    elif dist_thumb < 0.15 and dist_index < 0.15 and dist_middle < 0.15 and dist_ring < 0.15 and dist_pinky < 0.15:
        return "Closed Fist"

    elif dist_thumb > 0.16 and dist_index > 0.16 and dist_middle > 0.16 and dist_ring > 0.16 and dist_pinky > 0.16:
        return "Open Hand"

    else:
        return "Unknown Gesture"


def draw_radial_menu(frame, center, radius, options, selected_index=None, hovering_index=None):
    """Draws a radial menu centered at 'center' with the given options, highlighting the selected one."""
    angle_step = 360 / len(options)
    for i, option in enumerate(options):
        angle = np.radians(i * angle_step)
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] - radius * np.sin(angle))

        # Visual Effect for Hover
        if i == hovering_index:
            color = (0, 255, 0)  # Hover color (green)
            cv2.circle(frame, (x, y), 35, color, -1)  # Slightly bigger circle for hover effect
            cv2.circle(frame, (x, y), 35, (255, 255, 255), 3)  # White outline for visibility
        else:
            color = (200, 100, 50)  # Default color for non-hovered buttons
            cv2.circle(frame, (x, y), 30, color, -1)
            cv2.circle(frame, (x, y), 30, (255, 255, 255), 2)  # Add a subtle outline for all buttons

        # Visual Effect for Selected Option
        if i == selected_index:
            # "Pressed" effect (smaller circle, change color)
            cv2.circle(frame, (x, y), 30, (0, 0, 255), -1)  # Pressed effect (red color)
        
        # Display text centered within each circle
        cv2.putText(frame, option, (x - 20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw the radial menu center
    cv2.circle(frame, center, radius, (50, 200, 100), 2)


def detect_menu_selection(center, radius, cursor, options):
    """Detects which menu option the cursor is pointing at."""
    angle_step = 360 / len(options)
    for i, option in enumerate(options):
        angle = np.radians(i * angle_step)
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] - radius * np.sin(angle))
        if math.sqrt((cursor[0] - x) ** 2 + (cursor[1] - y) ** 2) < 30:  # Within option circle
            return i  # Return index of the selected option
    return None


# Variables for radial menu
menu_active = False
menu_center = None
menu_options = ["Play/Pause", "Previous","Next", "Scroll Up", "Scroll Down", "Mute", "Volume Up", "Volume Down"]
menu_radius = 100
menu_selection = None
selection_time = None  # Time when a selection starts
txt_time = None
selection_confirmed = False  # Flag to confirm selection
hovering_index = None  # To track hovered option
currentText = ""

# Access Webcam
cap = cv2.VideoCapture(0)

def map_to_screen(x, y):
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    # Define the usable range as a percentage of the screen
    usable_width = 0.8  # 90% of the screen width
    usable_height = 0.7 # 90% of the screen height
    
    # Adjust the ranges to leave some padding
    x_min, x_max = (1 - usable_width) / 2, 1 - (1 - usable_width) / 2
    y_min, y_max = (1 - usable_height) / 2, 1 - (0.5 - usable_height) / 2
    
    # Clamp hand positions to the adjusted range
    x_clamped = max(min(x, x_max), x_min)
    y_clamped = max(min(y, y_max), y_min)
    
    # Map clamped values to the screen coordinates
    screen_x = int((x_clamped - x_min) / (x_max - x_min) * screen_width)
    screen_y = int((y_clamped - y_min) / (y_max - y_min) * screen_height)
    
    return screen_x, screen_y


# Gesture state tracker
last_gesture = None
gesture_executed = False

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands_solution.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                gesture = classify_gesture(hand_landmarks)
                h, w, c = frame.shape
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                # Display the gesture name on the hand model
                
                #cv2.putText(frame, f"Gesture: {gesture}", (wrist_x + 20, wrist_y - 20),
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if gesture != last_gesture:
                    last_gesture = gesture
                    gesture_executed = False

                # Perform actions based on gesture
                if gesture == "Open Hand":
                    if not menu_active:  # Move cursor if menu is inactive
                        cursor_x, cursor_y = map_to_screen(wrist.x, wrist.y)
                        pyautogui.moveTo(cursor_x, cursor_y)
                    else:  # Navigate radial menu
                        cursor_x, cursor_y = wrist_x, wrist_y
                        hovering_index = detect_menu_selection(menu_center, menu_radius, (cursor_x, cursor_y), menu_options)
                        if hovering_index is not None:
                            menu_selection = menu_options[hovering_index]
                            # Start a timer to confirm selection
                            if selection_time is None:
                                selection_time = time.time()  # Record the start time of selection
                        else:
                            menu_selection = None
                            selection_time = None  # Reset timer if no selection

                elif gesture == "Victory Pose" and not menu_active:
                    # Activate the radial menu
                    menu_active = True
                    menu_center = (wrist_x, wrist_y)
                    #cv2.putText(frame, "Menu", (wrist_x + 20, wrist_y - 20),
                    #        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    currentText = "Menu"
                    
                
                elif gesture == "Closed Fist" and menu_active:
                    menu_active = False

                elif gesture == "Closed Fist" and not gesture_executed:
                    #pyautogui.click()
                    gesture_executed = True

                elif gesture == "Thumbs Up" and menu_active:
                    menu_active = False
                
                elif gesture == "Thumbs Up" and not gesture_executed:
                    pyautogui.click()
                    gesture_executed = True
                    currentText = "Click"

                elif gesture == "Pointing Up":
                    pyautogui.scroll(10)
                    cv2.putText(frame, "Scroll Up", (wrist_x + 20, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                
                
                elif gesture == "Pointing Pinky Finger":
                    pyautogui.scroll(-10)
                    cv2.putText(frame, "Scrolling Down", (wrist_x + 20, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                else:
                    # Cooldown time in seconds
                    COOLDOWN_TIME = 0.8
                    last_action_time = 0  # Track the time of the last action

                    # Get current time
                    current_time = time.time()

                    if gesture == "Chill" and not gesture_executed:
                        keyboard.send("play/pause media")
                        gesture_executed = True
                        currentText = "Play/Pause"
                    
                    if gesture == "Pointing Right" and not gesture_executed and current_time - last_action_time >= COOLDOWN_TIME:
                        keyboard.send("next track")
                        last_action_time = current_time
                        gesture_executed = True
                        currentText = "Next"

                    elif gesture == "Pointing Left" and not gesture_executed and current_time - last_action_time >= COOLDOWN_TIME:
                        keyboard.send("previous track")
                        last_action_time = current_time
                        gesture_executed = True
                        currentText = "Previous"

                
                if txt_time and time.time() - txt_time <= 1:
                    cv2.putText(frame, f"{currentText}", (wrist_x + 20, wrist_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)    
                else:
                    txt_time = time.time()
                    currentText = ""
                

                # Confirm the selection after 0.6 second of hovering
                if selection_time and time.time() - selection_time >= 0.6:
                    

                    if menu_selection:
                        # Add a label at the center of the radial menu
                        #cv2.putText(frame, f"Selected: {menu_selection}", (menu_center[0] - 50, menu_center[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                        #print(f"Confirmed Selection: {menu_selection}")
                        # Perform action based on selection
                        if menu_selection == "Play/Pause":
                            #pyautogui.press('pause')
                            keyboard.send("play/pause media")
                        elif menu_selection == "Previous": 
                            #pyautogui.hotkey('ctrl', 'r')
                            keyboard.send("previous track")
                        elif menu_selection == "Next": 
                            #pyautogui.hotkey('ctrl', 'r')
                            keyboard.send("next track")

                        elif menu_selection == "Scroll Up":
                            pyautogui.scroll(100)

                        elif menu_selection == "Scroll Down":
                            pyautogui.scroll(-100)

                        elif menu_selection == "Mute":
                            pyautogui.press('volumemute')

                        elif menu_selection == "Volume Up":
                            pyautogui.press('volumeup')

                        elif menu_selection == "Volume Down":
                            pyautogui.press('volumedown')
                    
                    
                    selection_time = None  # Reset timer after confirmation

        if menu_active:
            draw_radial_menu(frame, menu_center, menu_radius, menu_options, hovering_index=hovering_index)

        window_scale =  1.5  # Scale factor to enlarge the window (2x the original size)
        frame_resized = cv2.resize(frame, (int(frame.shape[1] * window_scale), int(frame.shape[0] * window_scale)))

        # Display the resized frame
        cv2.imshow('Hand Gesture Recognition with Radial Menu', frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 

finally:
    cap.release()
    cv2.destroyAllWindows()