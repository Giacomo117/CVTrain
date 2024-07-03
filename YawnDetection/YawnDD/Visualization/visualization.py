import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model = load_model('yawn_detection_model_mobilenet.h5')

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

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Make predictions
    predictions = model.predict(preprocessed_frame)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_label = get_class_label(predicted_class)

    # Display the result on the frame
    cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Yawn Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
