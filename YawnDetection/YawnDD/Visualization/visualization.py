import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model = load_model('yawn_detection_model_mobilenet.h5')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the frame for MobileNet
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0  # Normalize the frame
    return frame

# Function to get the class label
def get_class_label(prediction):
    if prediction == 0:
        return "Not Yawning"
    else:
        return "Yawning"

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Preprocess the face region
        preprocessed_face = preprocess_frame(face)

        # Make predictions on the face region
        predictions = model.predict(preprocessed_face)
        predicted_class = np.argmax(predictions, axis=1)[0]
        class_label = get_class_label(predicted_class)

        # Draw a rectangle around the face and display the result
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, class_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Yawn Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
