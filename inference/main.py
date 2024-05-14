import video as video
import cvdetection as cvdetection
import cv2
import torch
import numpy as np
import inference.util.utils as util
import inference.util.config as config
from typing import Generator, List
from detection.court_detector import CourtDetector
from detection.cameraestimator.CameraEstimator import CameraEstimator, estimate
from detection.teamclassifier.PlayerClustering import ColorHistogramClassifier
from detection.teamdetector.TeamDetector import TeamDetector
from detection.yolo_detector import YoloDetector
from detection.gameanalytics.GameAnalytics import GameAnalytics

def detect(): 

    # Create necessary detection objects
    court_detector: CourtDetector = CourtDetector(output_resolution=(1920, 1080))
    camera_estimator: CameraEstimator = CameraEstimator(output_resolution=(1920, 1080))
    
    team_classifier: ColorHistogramClassifier = ColorHistogramClassifier(num_of_teams=3)
    object_detector: YoloDetector = YoloDetector()
    team_detector: TeamDetector = TeamDetector(None, None)
    team_classifier.initialize_using_video('./videos/test.mp4', object_detector, court_detector, training_frames=50)
    team_detector.player_clustering = team_classifier

    analysis: GameAnalytics = GameAnalytics()
    analysis.infer_team_sides(
        './videos/test.mp4', court_detector, object_detector, team_classifier, camera_estimator, training_frames=1
    )

    # frame iterator to loop over input video frames
    frame_iterator: Generator[np.ndarray, None, None] = iter(video.generate_frames('./videos/test.mp4'))

    util.create_dir_if_not_exists(f'{util.get_project_root()}/{config.OUTPUT_DIR_NAME}')

    # output video writer
    output_video_path: str = f'{util.get_project_root()}/{config.OUTPUT_DIR_NAME}/{config.INPUT_VIDEO_NAME}'
    video_writer: cv2.VideoWriter = video.get_video_writer(output_video_path, None) 
    map2d = video.get_video_writer('map.mp4', None)

    frame_index: int = 0
    for frame in frame_iterator:
        # print(f'Processing {frame_index} - frame')
        

        drawn_frame: np.ndarray = frame.copy()
        masked_court_image, masked_edge_image = court_detector.get_masked_and_edge_court(drawn_frame)

        player_boxes, ball_boxes = object_detector.detect(frame, team_classifier)

        # get the estimated edge map
        estimated_edge_map = estimate(camera_estimator, masked_edge_image)

        # # detection.annotate(drawn_frame, person_detections)
        cvdetection.annotate(drawn_frame, player_boxes)
        cvdetection.annotate(drawn_frame, ball_boxes)

        output_frame = drawn_frame.copy()
        output_frame = cv2.addWeighted(
            src1=output_frame, 
            src2=estimated_edge_map,
            alpha=0.95, beta=0.9, gamma=0.)
        
        player_positions = team_detector.draw_on_field_points(output_frame, player_boxes)

        output_frame = cv2.addWeighted(
            src1=output_frame,
            src2=player_positions,
            alpha=0.95, beta=.9, gamma=0
        )
        
        analysis.update(camera_estimator.last_estimated_homography, player_boxes + ball_boxes)
        top_view_with_board = analysis.get_analytics()

        # video_writer.write(output_frame)
        video_writer.write(top_view_with_board)

        frame_index += 1
    
    analysis.save_coords_data(analysis.player_list, 'players.csv')
    video_writer.release()
    map2d.release()


if __name__ == '__main__':
    detect()