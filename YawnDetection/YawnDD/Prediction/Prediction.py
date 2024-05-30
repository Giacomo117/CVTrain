import tensorflow as tf

model = tf.keras.models.load_model('yawn_detection_model.h5')

import numpy as np

def predict_yawning(frame, model):
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized.astype('float32') / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    prediction = model.predict(frame_expanded)
    return np.argmax(prediction, axis=1)[0]

import cv2

# Define labels
labels = ['Normal', 'Talking', 'Yawning']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict yawning
    label_index = predict_yawning(frame, model)
    label = labels[label_index]

    # Overlay text on the frame
    cv2.putText(frame, f"Status: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Yawning Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
