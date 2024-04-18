import cv2
import mediapipe as mp
import logging
from mouse_control import move_mouse  # Import the mouse control function

# Configure logging to help in debugging
logging.basicConfig(level=logging.DEBUG)

# Initialize MediaPipe Hands module with settings optimized for real-time hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Non-static mode for video stream
    max_num_hands=1,  # Number of hands to detect
    min_detection_confidence=0.3,  # Confidence threshold for detection
    min_tracking_confidence=0.3)  # Confidence threshold for tracking

# Initialize MediaPipe drawing module to visualize landmarks
mp_drawing = mp.solutions.drawing_utils

# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

# Function definition here
def convert_to_screen_coords(hand_landmarks, screen_width, screen_height):
    """
    Convert hand landmarks to screen coordinates.
    
    Args:
    hand_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): Detected hand landmarks.
    screen_width (int): Width of the screen.
    screen_height (int): Height of the screen.

    Returns:
    tuple: Screen coordinates (x, y).
    """
    if not hand_landmarks:
        return None

    # Calculate the average of all x and y coordinates for simplicity
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]

    # Calculate the average position
    avg_x = sum(x_coords) / len(x_coords)
    avg_y = sum(y_coords) / len(y_coords)

    # Convert normalized coordinates to screen coordinates
    screen_x = int(avg_x * screen_width)
    screen_y = int(avg_y * screen_height)

    return (screen_x, screen_y)



# Main loop to continuously capture frames from the webcam
while cap.isOpened():
    success, image = cap.read()
    if not success:
        logging.warning("Ignoring empty camera frame.")
        continue  # Skip empty frames

    # Flip the image horizontally for a mirror view and convert the color from BGR to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the RGB image and track hands
    results = hands.process(image_rgb)

    # Check if any hands are detected and draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Convert hand landmark coordinates to screen coordinates
            screen_coords = convert_to_screen_coords(hand_landmarks, 1920, 1080)
            # Move the mouse cursor to the converted screen coordinates
            move_mouse(*screen_coords)

    # Display the processed image in a window named 'Hand Tracking'
    cv2.imshow('Hand Tracking', image)
    # Exit the loop when the 'ESC' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
