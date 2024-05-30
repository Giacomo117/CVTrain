import cv2
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import shutil

video_dir = './yawdd/YawDD dataset/Mirror/'
frame_output_dir = './annotated_frames/'

os.makedirs(os.path.join(frame_output_dir, 'Yawning'), exist_ok=True)
os.makedirs(os.path.join(frame_output_dir, 'No Yawning'), exist_ok=True)

def extract_frames(video_path, output_dir, yawning=False):
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    success, image = video_capture.read()
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}") as pbar:
        while success:
            yawning_suffix = '_Yawning' if yawning else '_No Yawning'
            class_dir = 'Yawning' if yawning else 'No Yawning'
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