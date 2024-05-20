import cv2
import numpy as np
from typing import Generator, Dict
from inference.util.containers import Rect, Color

# -------------------------------------------------------
# All useful methods related to video handling such as
# video config, getting frames and so on...
# -------------------------------------------------------


# -------------------------------------------------------
# Takes a video file path as input and returns a frame
# generator that can be used to iterate over frames of
# the video until the video ends.
# -------------------------------------------------------

class VideoHandler:
    
    def __init__(self, video_file_path: str):
        self.video: cv2.VideoCapture = cv2.VideoCapture(video_file_path)
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
    
    def generate_frames(self) -> Generator[np.ndarray, None, None]:

        print('FPS ====> ', self.video_fps)

        while self.video.isOpened():
            success: bool
            frame: cv2.typing.MatLike
            success, frame = self.video.read()

            if not success:
                break

            yield frame
        
        self.video.release()

    # -------------------------------------------------------
    # OpenCV video writer which is used to draw on frames and
    # then output that to our video with detections
    # -------------------------------------------------------
    def get_video_writer(self, output_video_path: str, frame_size: tuple) -> cv2.VideoWriter:
        
        if self.video_fps is None:
            print('None')
        
        return cv2.VideoWriter(
            output_video_path, 
            fourcc=cv2.VideoWriter_fourcc(*'vp80'),
            fps=self.video_fps,
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

