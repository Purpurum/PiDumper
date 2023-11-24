import cv2
import tempfile
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from io import BytesIO

def check_file_extension(file_path):
    filename, file_extension = os.path.splitext(file_path)
    return file_extension

def video_to_frames(video_file, start_time, end_time, skip_frames = 20):
    if type(video_file) != str:
        #filename = os.path.basename(video_file)
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_file.read())
        vidcap = cv2.VideoCapture(tfile.name)
    else:
        vidcap = cv2.VideoCapture(video_file)   
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_count = 0
    saved_frame_count = 0

    temp_dir = tempfile.mkdtemp()  # Create a temporary directory

    while True:
        ret, frame = vidcap.read()

        if not ret:
            break

        if start_frame <= frame_count <= end_frame and frame_count % skip_frames == 0:
            output_file = os.path.join(temp_dir, f"frame{saved_frame_count}--{frame_count}.jpg")
            cv2.imwrite(output_file, frame)
            saved_frame_count += 1

        frame_count += 1

    vidcap.release()

    return temp_dir, fps