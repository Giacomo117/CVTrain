import threading
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained models
yawn_model = load_model('C:/Users/39329/PycharmProjects/CVprova1/Final Union/Models/yawn_detection_model_mobilenet.h5')
eyes_model = load_model('C:/Users/39329/PycharmProjects/CVprova1/Final Union/Models/eyes_model.h5')

# Load the Haar Cascade classifier for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to preprocess the input frame for the yawn detection model
def preprocess_for_yawn(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    processed_frame = np.expand_dims(resized_frame, axis=0)
    processed_frame = processed_frame / 255.0
    return processed_frame

# Function to preprocess the input frame for the eye detection model
def preprocess_for_eyes(frame):
    resized_frame = cv2.resize(frame, (80, 80))
    processed_frame = np.expand_dims(resized_frame, axis=0)
    processed_frame = processed_frame / 255.0
    return processed_frame

# Function to get the class label for yawning
def get_yawn_class_label(prediction):
    return "Yawning" if prediction == 1 else "Not Yawning"

# Function to get the class label for eyes
def get_eye_class_label(prediction):
    return "Open" if prediction[0] < prediction[1] else "Closed"

# Shared resource locks
frame_lock = threading.Lock()
result_lock = threading.Lock()

# Shared resources
current_frame = None
faces_data = {}  # To store face_id: (face_rect, yawn_label)
eyes_data = {}   # To store face_id: [(eye_rect, eye_label), ...]
face_id_counter = 0

# Function to read frames from the webcam
def capture_frames():
    global current_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        with frame_lock:
            current_frame = frame.copy()

        # Display the frame
        with result_lock:
            if current_frame is not None:
                frame_display = current_frame.copy()
                for (x, y, w, h), yawn_label in faces_data.values():
                    # Calculate the radius for the rounded corners
                    radius = int(0.15 * min(w, h))

                    # Draw the four sides of the rectangle
                    cv2.line(frame_display, (x + radius, y), (x + w - radius, y), (55, 215, 255), 2)
                    cv2.line(frame_display, (x + radius, y + h), (x + w - radius, y + h), (55, 215, 255), 2)
                    cv2.line(frame_display, (x, y + radius), (x, y + h - radius), (55, 215, 255), 2)
                    cv2.line(frame_display, (x + w, y + radius), (x + w, y + h - radius), (55, 215, 255), 2)

                    # Draw the four rounded corners
                    cv2.ellipse(frame_display, (x + radius, y + radius), (radius, radius), 180, 0, 90, (55, 215, 255), 2)
                    cv2.ellipse(frame_display, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, (55, 215, 255), 2)
                    cv2.ellipse(frame_display, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, (55, 215, 255), 2)
                    cv2.ellipse(frame_display, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, (55, 215, 255), 2)
                    cv2.putText(frame_display, yawn_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (55, 215, 255), 2, cv2.LINE_AA)

                for eye_list in eyes_data.values():
                    for (ex, ey, ew, eh), eye_label in eye_list:
                        radius = int(0.15 * min(ew, eh))

                        # Top horizontal line
                        cv2.line(frame_display, (ex + radius, ey), (ex + ew - radius, ey), (200,200,200), 2)
                        # Bottom horizontal line
                        cv2.line(frame_display, (ex + radius, ey + eh), (ex + ew - radius, ey + eh), (200,200,200), 2)
                        # Left vertical line
                        cv2.line(frame_display, (ex, ey + radius), (ex, ey + eh - radius), (200,200,200), 2)
                        # Right vertical line
                        cv2.line(frame_display, (ex + ew, ey + radius), (ex + ew, ey + eh - radius), (200,200,200), 2)

                        # Top-left corner
                        cv2.ellipse(frame_display, (ex + radius, ey + radius), (radius, radius), 180, 0, 90, (200,200,200), 2)
                        # Top-right corner
                        cv2.ellipse(frame_display, (ex + ew - radius, ey + radius), (radius, radius), 270, 0, 90, (200,200,200), 2)
                        # Bottom-left corner
                        cv2.ellipse(frame_display, (ex + radius, ey + eh - radius), (radius, radius), 90, 0, 90, (200,200,200), 2)
                        # Bottom-right corner
                        cv2.ellipse(frame_display, (ex + ew - radius, ey + eh - radius), (radius, radius), 0, 0, 90, (200,200,200), 2)
                        cv2.putText(frame_display, eye_label, (ex, ey-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200,200,200), 2, cv2.LINE_AA)

                cv2.imshow('Yawn and Eyes Detection', frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to process frames
def process_frames():
    global faces_data, eyes_data, face_id_counter

    while True:
        time.sleep(0.05)  # Adjust the sleep time for balancing performance
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        new_faces_data = {}
        new_eyes_data = {}

        for (x, y, w, h) in faces:
            face_id_counter += 1
            face_id = face_id_counter
            face = frame[y:y+h, x:x+w]
            preprocessed_face_yawn = preprocess_for_yawn(face)
            yawn_predictions = yawn_model.predict(preprocessed_face_yawn)
            yawn_class = np.argmax(yawn_predictions, axis=1)[0]
            yawn_label = get_yawn_class_label(yawn_class)

            new_faces_data[face_id] = ((x, y, w, h), yawn_label)

            # Detect eyes within the face region
            face_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

            eye_list = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Limit to first 2 detected eyes
                eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
                preprocessed_eye = preprocess_for_eyes(eye)
                prediction = eyes_model.predict(preprocessed_eye)
                eye_label = get_eye_class_label(prediction[0])

                eye_list.append(((x+ex, y+ey, ew, eh), eye_label))

            new_eyes_data[face_id] = eye_list

        with result_lock:
            faces_data = new_faces_data
            eyes_data = new_eyes_data

# Start the threads
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)

capture_thread.start()
process_thread.start()

capture_thread.join()
process_thread.join()
