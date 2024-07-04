import cv2
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

video_dir = './yawdd/YawDD dataset/Mirror/'
frame_output_dir = './annotated_frames/'

os.makedirs(os.path.join(frame_output_dir, 'Yawning'), exist_ok=True)
os.makedirs(os.path.join(frame_output_dir, 'No Yawning'), exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read file filenames.txt
with open('./Training/filenames.txt', 'r') as file:
    filenames = file.readlines()

# Create a dictionary to store the frames and their classes
frames_dict = {}
for line in filenames:
    parts = line.split('_frame_')
    video_name = parts[0].strip()
    frame_number = int(parts[1].split('_')[0])
    yawning = '_Yawning' in parts[1]
    if video_name not in frames_dict:
        frames_dict[video_name] = []
    frames_dict[video_name].append((frame_number, yawning))

def extract_frames(video_path, output_dir, yawning=False):
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    success, image = video_capture.read()
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}") as pbar:
        while success:
            yawning_suffix = "_No Yawning"
            class_dir = "No Yawning"
            if yawning and os.path.basename(video_path) in frames_dict:
                if count in [frame[0] for frame in frames_dict[os.path.basename(video_path)]]:
                    yawning_suffix = "_Yawning"
                    class_dir = "Yawning"

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,  # Increase scale factor to make detection more robust
                minNeighbors=5,
                minSize=(30, 30),  # Increase min size to avoid very small face detections
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # If no face is detected, save the full frame or skip saving
            if len(faces) == 0:
                output_filepath = os.path.join(output_dir, class_dir, f"{os.path.basename(video_path)}_frame_{count}{yawning_suffix}.jpg")
                cv2.imwrite(output_filepath, image)
            else:
                for (x, y, w, h) in faces:
                    # Ensure the detected face is a reasonable size
                    if w > 50 and h > 50:  # Only consider faces larger than 50x50 pixels
                        face = image[y:y+h, x:x+w]
                        output_filepath = os.path.join(output_dir, class_dir, f"{os.path.basename(video_path)}_frame_{count}{yawning_suffix}.jpg")
                        cv2.imwrite(output_filepath, face)
                        break  # Save only the first detected face

            success, image = video_capture.read()
            count += 1
            pbar.update(1)
    video_capture.release()

# Process videos in parallel using ThreadPoolExecutor
# with ThreadPoolExecutor(max_workers=4) as executor:
#     for gender_dir in ['Female_mirror', 'Male_mirror Avi Videos']:
#         gender_dir_path = os.path.join(video_dir, gender_dir)
#         for video_file in os.listdir(gender_dir_path):
#             yawning = 'Yawning' in video_file
#             executor.submit(extract_frames, os.path.join(gender_dir_path, video_file), frame_output_dir, yawning)


for gender_dir in ['Female_mirror', 'Male_mirror Avi Videos']:
    gender_dir_path = os.path.join(video_dir, gender_dir)
    for video_file in os.listdir(gender_dir_path):
        yawning = 'Yawning' in video_file
        extract_frames(os.path.join(gender_dir_path, video_file), frame_output_dir, yawning)

print("Done!")
