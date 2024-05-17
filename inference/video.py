import cv2
import numpy as np
import os
from typing import Generator, Dict
from util.containers import Rect, Color


# -------------------------------------------------------
# All useful methods related to video handling such as
# video config, getting frames and so on...
# -------------------------------------------------------


# -------------------------------------------------------
# Takes a video file path as input and returns a frame
# generator that can be used to iterate over frames of
# the video until the video ends.
# -------------------------------------------------------
def generate_frames(video_file_path: str) -> Generator[np.ndarray, None, None]:
    video: cv2.VideoCapture = cv2.VideoCapture(video_file_path)

    while video.isOpened():
        success: bool
        frame: cv2.typing.MatLike
        success, frame = video.read()

        if not success:
            break

        yield frame
    
    video.release()

# -------------------------------------------------------
# OpenCV video writer which is used to draw on frames and
# then output that to our video with detections
# -------------------------------------------------------
def get_video_writer(output_video_path: str, frame_size: tuple) -> cv2.VideoWriter:
    
    return cv2.VideoWriter(
        output_video_path, 
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=60,
        frameSize=frame_size,
        isColor=True
    )

# -------------------------------------------------------
# Drawing utilities on videos such as drawing rects, etc.
# -------------------------------------------------------

def draw_rect(image: np.ndarray, coords: tuple, color: tuple, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, (coords[0], coords[1]), (coords[2], coords[3]), color, thickness)
    return image


def draw_filled_rect(image: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, -1)
    return image

