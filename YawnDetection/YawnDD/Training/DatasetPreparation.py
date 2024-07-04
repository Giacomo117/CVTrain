import cv2
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

video_dir = './yawdd/YawDD dataset/Mirror/'
frame_output_dir = './annotated_frames/'

os.makedirs(os.path.join(frame_output_dir, 'Yawning'), exist_ok=True)
os.makedirs(os.path.join(frame_output_dir, 'No Yawning'), exist_ok=True)

# Read file filenames.txt
with open('./Training/filenames.txt', 'r') as file:
    filenames = file.readlines()

# Create a dictionary to store the frames and their classes
frames_dict = {}
for line in filenames:
    parts = line.strip().split('_frame_')
    video_name = parts[0].strip()  # Trim any whitespace
    frame_number = int(parts[1].split('_')[0])
    yawning = '_Yawning' in parts[1]
    if video_name not in frames_dict:
        frames_dict[video_name] = {}
    frames_dict[video_name][frame_number] = yawning

# Print statistics
print(f"Number of videos: {len(frames_dict.keys())}")
total_frames = sum(len(frames) for frames in frames_dict.values())
print(f"Total number of video frames: {total_frames}")


global totalProcessed
global yawnsCounted
totalProcessed = 0
yawnsCounted = 0

def extract_frames(video_path, output_dir):
    global totalProcessed
    global yawnsCounted 
    video_name = os.path.basename(video_path).strip()
    if video_name not in frames_dict:
        print(f"Video {video_name} not found in frames_dict")
        return

    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    success, image = video_capture.read()
    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while success:
            class_dir = "No Yawning"
            if count in frames_dict[video_name]:
                if frames_dict[video_name][count]:
                    class_dir = "Yawning"
                # Debug print
                print(f"Video: {video_name}, Frame: {count}, Class: {class_dir}")

            output_filepath = os.path.join(output_dir, class_dir, f"{video_name}_frame_{count}.jpg")
            cv2.imwrite(output_filepath, image)
            
            count += 1
            totalProcessed += 1
            yawnsCounted += 1 if class_dir == "Yawning" else 0
            pbar.update(1)
            success, image = video_capture.read()
            if(not success and count < total_frames):
                print("Error reading frame", count, "from", video_name)
                while not success and count < total_frames:
                    count += 1
                    success, image = video_capture.read()
                    print("Trying frame", count)
            
    video_capture.release()

# with ThreadPoolExecutor(max_workers=4) as executor:
#     for gender_dir in ['Female_mirror', 'Male_mirror Avi Videos']:
#         gender_dir_path = os.path.join(video_dir, gender_dir)
#         for video_file in os.listdir(gender_dir_path):
#             video_path = os.path.join(gender_dir_path, video_file)
#             executor.submit(extract_frames, video_path, frame_output_dir)
for gender_dir in ['Female_mirror', 'Male_mirror Avi Videos']:
    gender_dir_path = os.path.join(video_dir, gender_dir)
    for video_file in os.listdir(gender_dir_path):
        video_path = os.path.join(gender_dir_path, video_file)
        extract_frames(video_path, frame_output_dir)
print("\n\nDone!", totalProcessed, "frames processed,", yawnsCounted, "yawns counted")

