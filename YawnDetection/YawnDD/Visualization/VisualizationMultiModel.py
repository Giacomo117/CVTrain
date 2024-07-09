import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained models
model_gray = load_model('yawn_detection_model_mobilenet_grayscale.h5')
model_color = load_model('yawn_detection_model_mobilenet.h5')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the frame for MobileNet
def preprocess_frame(frame, is_grayscale=False):
    frame = cv2.resize(frame, (64, 64))
    if is_grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, axis=-1)  # Add a single channel dimension
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = frame / 255.0  # Normalize the frame
    return frame

# Function to get the class label
def get_class_label(prediction):
    return "Yawning" if prediction == 1 else "Not Yawning"

# Function to draw rounded rectangle
def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness, radius=10):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

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

        # Preprocess the face region for the grayscale model
        preprocessed_face_gray = preprocess_frame(face, is_grayscale=True)

        # Preprocess the face region for the color model
        preprocessed_face_color = preprocess_frame(face, is_grayscale=False)

        # Make predictions on the face region
        predictions_gray = model_gray.predict(preprocessed_face_gray)
        predicted_class_gray = np.argmax(predictions_gray, axis=1)[0]
        class_label_gray = get_class_label(predicted_class_gray)

        predictions_color = model_color.predict(preprocessed_face_color)
        predicted_class_color = np.argmax(predictions_color, axis=1)[0]
        class_label_color = get_class_label(predicted_class_color)

        # Draw a rounded rectangle around the face
        draw_rounded_rectangle(frame, (x, y), (x+w, y+h), (252, 226, 5), 2)

        # Display the result for the grayscale model
        cv2.putText(frame, class_label_gray, (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "(Grayscale)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the result for the color model in rainbow colors
        colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 127, 255), (0, 0, 255), (139, 0, 255)]
        for i, char in enumerate(class_label_color):
            color = colors[i % len(colors)]
            cv2.putText(frame, char, (x, y+h+30+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "(Color)", (x, y+h+30+len(class_label_color)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Yawn Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
