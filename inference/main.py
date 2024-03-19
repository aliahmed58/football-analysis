import video as video
import cvdetection as cvdetection
import cv2
import torch
import numpy as np
import inference.util.utils as util
import inference.util.config as config
from typing import Generator, List
from detection.court_detector import CourtDetector

def detect(): 

    # Create necessary detection objects
    court_detector = CourtDetector(output_resolution=(1920, 1080))

    # load the yolov5 model for this project
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/best.pt',device=0, force_reload=True)
    # model = YOLO('yolov8n.pt', task='predict', verbose=True)
    # frame iterator to loop over input video frames
    frame_iterator: Generator[np.ndarray, None, None] = iter(video.generate_frames('./videos/test.mp4'))

    util.create_dir_if_not_exists(f'{util.get_project_root()}/{config.OUTPUT_DIR_NAME}')

    # output video writer
    output_video_path: str = f'{util.get_project_root()}/{config.OUTPUT_DIR_NAME}/{config.INPUT_VIDEO_NAME}'
    video_writer: cv2.VideoWriter = video.get_video_writer(output_video_path, None) 

    frame_index: int = 0
    for frame in frame_iterator:
        print(f'Processing {frame_index} - frame')
        
        # do detections
        results = model(frame, size=1280)
        
        # get detections from the predicted results
        detections = cvdetection.Detection.get_detections_from_results(
            predictions=results.pred[0].cpu().numpy(),
            names=model.names
        )

        # filter detections by class names
        ball_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="ball")
        referee_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="referee")
        goalkeeper_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="goalkeeper")
        player_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="player")

        person_detections: List[cvdetection.Detection] = referee_detections + goalkeeper_detections + player_detections

        drawn_frame: np.ndarray = frame.copy()

        masked_court_image, masked_edge_image = court_detector.get_masked_and_edge_court(frame)

        if frame_index == 0:
            print(type(masked_court_image))
        # detection.annotate(drawn_frame, person_detections)
        cvdetection.annotate(drawn_frame, ball_detections)
        cvdetection.annotate(drawn_frame, person_detections)

        video_writer.write(masked_edge_image)
        # send detected data to the court detector ?


        # plot x,y coordinates to 2d field 

        # emit this data to csv file
        frame_index += 1
    
    video_writer.release()


if __name__ == '__main__':
    detect()