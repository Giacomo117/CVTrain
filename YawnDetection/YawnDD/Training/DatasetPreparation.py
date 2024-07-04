import cv2
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

video_dir = './yawdd/YawDD dataset/Mirror/'
frame_output_dir = './annotated_frames/'

os.makedirs(os.path.join(frame_output_dir, 'Yawning'), exist_ok=True)
os.makedirs(os.path.join(frame_output_dir, 'No Yawning'), exist_ok=True)

# read file filenames.txt
with open('./Training/filenames.txt', 'r') as file:
    filenames = file.readlines()
# Now, each line is a string, where there's a first part with the video name, then "_frame_", and a number, and then "_Yawning" or "_No Yawning"
# we need to create a dictionary where the key is the video name, and the value is a list of tuples with the frame number and the class

# create a dictionary to store the frames and their classes
frames_dict = {}
for line in filenames:
    # split the line in the parts
    parts = line.split('_frame_')
    # get the video name
    video_name = parts[0]
    # get the frame number
    frame_number = int(parts[1].split('_')[0])
    # get the class
    yawning = True
    # if the video name is not in the dictionary, add it
    if video_name not in frames_dict:
        frames_dict[video_name] = []
    # add the frame number and the class to the list
    frames_dict[video_name].append((frame_number))

def extract_frames(video_path, output_dir, yawning=False):
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    success, image = video_capture.read()
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}") as pbar:
        while success:
            yawning_suffix = "_No Yawning"
            class_dir = "No Yawning"
            if yawning:
                # check if we find the video name in the dictionary
                if os.path.basename(video_path) in frames_dict:
                    # check if the frame number is in the list
                    if count in frames_dict[os.path.basename(video_path)]:
                        yawning_suffix = "_Yawning"
                        class_dir = "Yawning"

            
            cv2.imwrite(f"{output_dir}/{class_dir}/{os.path.basename(video_path)}_frame_{count}{yawning_suffix}.jpg", image)
            success, image = video_capture.read()
            count += 1
            pbar.update(1)

with ThreadPoolExecutor(max_workers=4) as executor:
    for gender_dir in ['Female_mirror', 'Male_mirror Avi Videos']:
        gender_dir_path = os.path.join(video_dir, gender_dir)
        for video_file in os.listdir(gender_dir_path):
            yawning = 'Yawning' in video_file
            executor.submit(extract_frames, os.path.join(gender_dir_path, video_file), frame_output_dir, yawning)