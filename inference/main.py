import video
import detection
import cv2
import torch
import numpy as np
from typing import Generator, List

def detect(): 
    # load the yolov5 model for this project
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/best.pt', force_reload=True)

    # frame iterator to loop over input video frames
    frame_iterator: Generator[np.ndarray, None, None] = iter(video.generate_frames('./videos/test.mp4'))

    # output video writer
    video_writer: cv2.VideoWriter = video.get_video_writer('./out.mp4', None) 

    frame_index = 0
    for frame in frame_iterator:
        print(f'Processing {frame_index} - frame')
        # do detections
        results = model(frame, size=1280)

        # get detections from the predicted results
        detections = detection.Detection.get_detections_from_results(
            predictions=results.pred[0].cpu().numpy(),
            names=model.names
        )

        # filter detections by class names
        ball_detections: List[detection.Detection] = detection.filter_detections_by_class(detections=detections, class_name="ball")
        referee_detections: List[detection.Detection] = detection.filter_detections_by_class(detections=detections, class_name="referee")
        goalkeeper_detections: List[detection.Detection] = detection.filter_detections_by_class(detections=detections, class_name="goalkeeper")
        player_detections: List[detection.Detection] = detection.filter_detections_by_class(detections=detections, class_name="player")

        person_detections: List[detection.Detection] = referee_detections + goalkeeper_detections + player_detections

        drawn_frame: np.ndarray = frame.copy()
        detection.annotate(drawn_frame, person_detections)
        detection.annotate(drawn_frame, ball_detections)
        
        video_writer.write(drawn_frame)

        # send detected data to the court detector ?


        # plot x,y coordinates to 2d field 

        # emit this data to csv file
        frame_index += 1
    
    video_writer.release()


if __name__ == '__main__':
    detect()