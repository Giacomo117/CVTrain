import threading
import time
from collections import deque
import torch.nn as nn
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import torch
from torchvision.models import resnet50
import math
import json

# Global variables for circular buffer
log_buffer = deque(maxlen=450)  # Assuming 30 frames per second

# Load the trained models
yawn_model = load_model('Models/yawn_detection_model_mobilenet.h5')
eyes_model = load_model('Models/eyes_model.h5')

model = resnet50(weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = True

# Change the final layer
model.fc = nn.Linear(in_features=2048, out_features=136)

model = model.to('cpu')

# Load the model checkpoint
checkpoint = torch.load("Models/model.pth",
                        map_location=torch.device('cpu'))
# Load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])

# Execute model evaluation
model.eval()

# Load the Haar Cascade classifier for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


logging_file_path = 'detection_log.txt'
json_logging_file_path = 'detection_log.json'
should_terminate = False
metrics_display = ""

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
keypoints_data = {}  # To store face_id: keypoints
face_id_counter = 0

# Function to add log entries to the circular buffer
def add_log_to_buffer(log_data):
    timestamp = time.time()
    log_buffer.append((timestamp, log_data))

# Function to remove old entries
def remove_old_entries():
    current_time = time.time()
    while log_buffer and (current_time - log_buffer[0][0]) > 15:
        log_buffer.popleft()

# Function to read frames from the webcam
def log_face_details(face_id, x, y, w, h, yawn_label):
    log_message = f"Face ID: {face_id}\n"
    log_message += f"  - Face Position: (x: {x}, y: {y}, w: {w}, h: {h})\n"
    log_message += f"  - Yawning: {yawn_label}\n"

    with open(logging_file_path, 'a') as log_file:
        log_file.write(log_message)
        log_file.write("\n")

    log_data = {
        "Face ID": face_id,
        "Face Position": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        "Yawning": yawn_label,
    }
    with open(json_logging_file_path, 'a') as json_log_file:
        json.dump(log_data, json_log_file)
        json_log_file.write("\n")

    add_log_to_buffer(log_data)
    remove_old_entries()

def log_eye_details(face_id, eye_list):
    log_message = f"Face ID: {face_id}\n"
    log_message += f"  - Eyes:\n"
    if eye_list:
        for (ex, ey, ew, eh), eye_label in eye_list:
            log_message += f"    - Eye Position: (x: {ex}, y: {ey}, w: {ew}, h: {eh}), Label: {eye_label}\n"
    else:
        log_message += "    - No eyes detected, considered closed\n"
    if len(eye_list) == 1:
        log_message += "    - One eye detected, considered the other as closed\n"

    with open(logging_file_path, 'a') as log_file:
        log_file.write(log_message)
        log_file.write("\n")

    log_data = {
        "Face ID": face_id,
        "Eyes": []
    }
    if eye_list:
        for (ex, ey, ew, eh), eye_label in eye_list:
            log_data["Eyes"].append({
                "Eye Position": {"x": int(ex), "y": int(ey), "w": int(ew), "h": int(eh)},
                "Label": eye_label
            })
    else:
        log_data["Eyes"].append("No eyes detected, considered closed")
    if len(eye_list) == 1:
        log_data["Eyes"].append("One eye detected, considered the other as closed")
    with open(json_logging_file_path, 'a') as json_log_file:
        json.dump(log_data, json_log_file)
        json_log_file.write("\n")

    add_log_to_buffer(log_data)
    remove_old_entries()

def log_keypoints_details(face_id, keypoints):
    log_message = f"Face ID: {face_id}\n"
    log_message += f"  - Keypoints:\n"
    if len(keypoints) > 0:
        # for (kx, ky) in keypoints:
        #     log_message += f"    - Keypoint Position: (x: {kx}, y: {ky})\n"
        eye_to_eye_distance = np.linalg.norm(np.array(keypoints[42]) - np.array(keypoints[39]))
        nose_to_chin_distance = np.linalg.norm(np.array(keypoints[33]) - np.array(keypoints[8]))
        log_message += f"    - Eye-to-eye distance: {eye_to_eye_distance:.2f}\n"
        log_message += f"    - Nose-to-chin distance: {nose_to_chin_distance:.2f}\n"
        
        #compute face rotation angle
        left_eye_center = np.array(keypoints[39])
        right_eye_center = np.array(keypoints[42])
        face_rotation_angle = np.arctan((right_eye_center[1] - left_eye_center[1]) / (right_eye_center[0] - left_eye_center[0]))
        log_message += f"    - Face rotation angle: {face_rotation_angle:.2f}\n"
        
    else:
        log_message += "    - No keypoints detected\n"

    with open(logging_file_path, 'a') as log_file:
        log_file.write(log_message)
        log_file.write("\n")

    log_data = {
        "Face ID": face_id,
        "Keypoints": []
    }
    if len(keypoints) > 0:
        # for (kx, ky) in keypoints:
        #     log_data["Keypoints"].append({"x": float(kx), "y": float(ky)})
        eye_to_eye_distance = float(np.linalg.norm(np.array(keypoints[42]) - np.array(keypoints[39])))
        nose_to_chin_distance = float(np.linalg.norm(np.array(keypoints[33]) - np.array(keypoints[8])))
        # add face rotation angle to log data
        left_eye_center = np.array(keypoints[39])
        right_eye_center = np.array(keypoints[42])
        face_rotation_angle = np.arctan((right_eye_center[1] - left_eye_center[1]) / (right_eye_center[0] - left_eye_center[0]))
        # convert the face rotation angle to a float
        face_rotation_angle = float(face_rotation_angle)
        log_data["Geometrics"] = {
            "Eye-to-eye distance": eye_to_eye_distance,
            "Nose-to-chin distance": nose_to_chin_distance,
            "Face rotation angle": face_rotation_angle
        }
    else:
        log_data["Keypoints"].append("No keypoints detected")
    with open(json_logging_file_path, 'a') as json_log_file:
        json.dump(log_data, json_log_file)
        json_log_file.write("\n")

    add_log_to_buffer(log_data)
    remove_old_entries()

def process_face(faces, frame):
    global faces_data, face_id_counter
    new_faces_data = {}
    for (x, y, w, h) in faces:
        face_id_counter += 1
        face_id = face_id_counter
        face = frame[y:y+h, x:x+w]
        preprocessed_face_yawn = preprocess_for_yawn(face)
        yawn_predictions = yawn_model.predict(preprocessed_face_yawn)
        yawn_class = np.argmax(yawn_predictions, axis=1)[0]
        yawn_label = get_yawn_class_label(yawn_class)

        new_faces_data[face_id] = ((x, y, w, h), yawn_label)
        log_face_details(face_id, x, y, w, h, yawn_label)
    
    with result_lock:
        faces_data = new_faces_data

def process_eyes(faces, frame, gray):
    global eyes_data, face_id_counter
    new_eyes_data = {}
    for (x, y, w, h) in faces:
        face_id_counter += 1
        face_id = face_id_counter
        face = frame[y:y+h, x:x+w]

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
        log_eye_details(face_id, eye_list)

    with result_lock:
        eyes_data = new_eyes_data

def process_keypoints(faces, frame):
    global keypoints_data, face_id_counter
    new_keypoints_data = {}
    for (x, y, w, h) in faces:
        face_id_counter += 1
        face_id = face_id_counter
        cropped_image = frame[y:y+h, x:x+w]
        image = cropped_image.copy()
        if image.shape[0] > 1 and image.shape[1] > 1:
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float)
            image = image.unsqueeze(0).to('cpu')

            outputs = model(image)
            outputs = outputs.cpu().detach().numpy()
            outputs = outputs.reshape(-1, 2)
            keypoints = outputs

            for i, p in enumerate(keypoints):
                p[0] = p[0] / 224 * cropped_image.shape[1]
                p[1] = p[1] / 224 * cropped_image.shape[0]
                p[0] += x
                p[1] += y

            new_keypoints_data[face_id] = keypoints
            log_keypoints_details(face_id, keypoints)
    
    with result_lock:
        keypoints_data = new_keypoints_data

# Function to read frames from the webcam
def capture_frames():
    global current_frame, should_terminate
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while not should_terminate:
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

                    # Draw the four corners of the rectangle
                    cv2.ellipse(frame_display, (x + radius, y + radius), (radius, radius), 180.0, 0, 90, (55, 215, 255), 2)
                    cv2.ellipse(frame_display, (x + w - radius, y + radius), (radius, radius), 270.0, 0, 90, (55, 215, 255), 2)
                    cv2.ellipse(frame_display, (x + radius, y + h - radius), (radius, radius), 90.0, 0, 90, (55, 215, 255), 2)
                    cv2.ellipse(frame_display, (x + w - radius, y + h - radius), (radius, radius), 0.0, 0, 90, (55, 215, 255), 2)

                    # Display the yawn label
                    cv2.putText(frame_display, f"Yawning: {yawn_label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                for eye_list in eyes_data.values():
                    for (ex, ey, ew, eh), eye_label in eye_list:
                        cv2.rectangle(frame_display, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                        cv2.putText(frame_display, f"{eye_label}", (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                for keypoints in keypoints_data.values():
                    for (kx, ky) in keypoints:
                        cv2.circle(frame_display, (int(kx), int(ky)), 2, (0, 0, 255), -1)

                cv2.imshow('Webcam Feed', frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            should_terminate = True

    cap.release()
    cv2.destroyAllWindows()

# Function to retrieve the last 15 seconds of logs
def get_recent_logs():
    with result_lock:
        return list(log_buffer)

# Periodically call this function to clean up old log entries
def periodic_cleanup():
    while not should_terminate:
        time.sleep(1)
        remove_old_entries()

# Function to process frames
def process_frames():
    global faces_data, eyes_data, keypoints_data, face_id_counter, should_terminate

    while not should_terminate:
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Sort faces based on area (w * h) in descending order and select the largest one
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            largest_face = faces[0:1]  # Take only the largest face
        else:
            largest_face = []

        face_thread = threading.Thread(target=process_face, args=(largest_face, frame))
        eye_thread = threading.Thread(target=process_eyes, args=(largest_face, frame, gray))
        keypoints_thread = threading.Thread(target=process_keypoints, args=(largest_face, frame))

        face_thread.start()
        eye_thread.start()
        keypoints_thread.start()

        face_thread.join()
        eye_thread.join()
        keypoints_thread.join()


def compute_metrics():
    global metrics_display, should_terminate
    while not should_terminate:
        time.sleep(1)
        # Ensure the buffer isn't empty
        if not log_buffer:
            continue

        with result_lock:
            buffer_snapshot = list(log_buffer)

        # Initialize metrics
        total_yawns = 0
        total_eye_open = 0
        total_eye_closed = 0
        relative_rotation_angle = 0.0
        total_rotation_angle = 0.0

        num_faces = 0
        num_yawn_entries = 0
        num_eye_entries = 0
        num_rotation_entries = 0

        first_rotation_angle = None

        for timestamp, log_entry in buffer_snapshot:
            if "Yawning" in log_entry:
                num_faces += 1
                if log_entry["Yawning"] == "Yawning":
                    total_yawns += 1
                num_yawn_entries += 1

            if "Eyes" in log_entry:
                num_faces += 1
                # Track if both eyes are open in this log entry
                both_eyes_open = True
                for eye in log_entry["Eyes"]:
                    if isinstance(eye, dict):
                        if "Label" not in eye or eye["Label"] == "Closed":
                            both_eyes_open = False
                            break
                    else:
                        both_eyes_open = False
                        break

                # if there's just one eye, or less than two eyes, consider the other eye closed
                if len(log_entry["Eyes"]) < 2:
                    both_eyes_open = False
                
                if both_eyes_open:
                    total_eye_open += 1
                else:
                    total_eye_closed += 1
                num_eye_entries += 1

            if "Geometrics" in log_entry and "Face rotation angle" in log_entry["Geometrics"]:
                num_faces += 1
                face_rotation_angle = log_entry["Geometrics"]["Face rotation angle"]
                if first_rotation_angle is None:
                    first_rotation_angle = face_rotation_angle
                total_rotation_angle += abs(face_rotation_angle)
                relative_rotation_angle += face_rotation_angle
                num_rotation_entries += 1

        if num_faces > 0:
            avg_yawn_rate = total_yawns / num_yawn_entries if num_yawn_entries > 0 else 0
            avg_eye_open_rate = total_eye_open / num_eye_entries if num_eye_entries > 0 else 0
            avg_rotation_angle = relative_rotation_angle / num_rotation_entries if num_rotation_entries > 0 else 0

            if first_rotation_angle is not None:
                relative_rotation_angle = abs(relative_rotation_angle - first_rotation_angle)
            else:
                relative_rotation_angle = 0
        else:
            avg_yawn_rate = 0
            avg_eye_open_rate = 0

        # Handle possible NaN or None values
        avg_yawn_rate = avg_yawn_rate if not math.isnan(avg_yawn_rate) else 0
        avg_eye_open_rate = avg_eye_open_rate if not math.isnan(avg_eye_open_rate) else 0
        relative_rotation_angle = relative_rotation_angle if not math.isnan(relative_rotation_angle) else 0

        # Prepare the metrics display string
        print(f"Avg Yawn Rate: {avg_yawn_rate:.2f}, Avg Eye Open Rate: {avg_eye_open_rate:.2f}, Avg Face Rotation Angle: {relative_rotation_angle:.2f}")

        # Determine the display message based on metrics
        if (avg_eye_open_rate < 0.45) or avg_yawn_rate > 0.85 or relative_rotation_angle > 9.00:
            metrics_display = "Driver is drowsy"
        elif relative_rotation_angle > 0.15 and avg_yawn_rate > 0.3 and avg_eye_open_rate < 0.5:
            metrics_display = "Driver may be falling asleep"
        elif total_rotation_angle > 0.5 and relative_rotation_angle < 0.1 and avg_yawn_rate < 0.3 and avg_eye_open_rate > 0.8:
            metrics_display = "Driver is focused"
        elif avg_eye_open_rate < 0.5:
            metrics_display = "Driver may be falling asleep"
        else:
            metrics_display = ""


# Function to display the frame with alerts
def display_frame():
    global metrics_display, should_terminate
    while not should_terminate:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                continue

        # Add the metrics display to the frame
        if metrics_display:
            cv2.putText(frame, metrics_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Driver Monitoring System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Create threads for capturing and processing frames
capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
metrics_thread = threading.Thread(target=compute_metrics)
display_thread = threading.Thread(target=display_frame)

# Start the threads
capture_thread.start()
process_thread.start()
metrics_thread.start()
display_thread.start()

# Wait for the threads to finish (they won't in this case)
capture_thread.join()
process_thread.join()
metrics_thread.join()
display_thread.join()

# close the log file, adding the missin parenthesis in the json
with open(json_logging_file_path, 'a') as json_log_file:
    json_log_file.write("}")
