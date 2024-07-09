import threading
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained models
yawn_model = load_model('./Models/yawn_detection_model_mobilenet.h5')
eyes_model = load_model('./Models/eyes_model.h5')

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
                    cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_display, yawn_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

                for eye_list in eyes_data.values():
                    for (ex, ey, ew, eh), eye_label in eye_list:
                        cv2.rectangle(frame_display, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                        cv2.putText(frame_display, eye_label, (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

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
