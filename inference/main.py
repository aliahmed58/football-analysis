from inference.video import VideoHandler
import inference.cvdetection as cvdetection
import cv2
import torch
import numpy as np
import sqlalchemy as sa
import inference.util.utils as util
import inference.util.config as config
import inference.persistance.persist as db
from typing import Generator
from torch.multiprocessing import set_start_method
from inference.detection.court_detector import CourtDetector
from inference.detection.cameraestimator.CameraEstimator import CameraEstimator, estimate
from inference.detection.teamclassifier.PlayerClustering import ColorHistogramClassifier
from inference.detection.teamdetector.TeamDetector import TeamDetector
from inference.detection.yolo_detector import YoloDetector
from inference.detection.gameanalytics.GameAnalytics import GameAnalytics

engine: sa.Engine = db.get_engine()
object_detector = None

def detect(input_video_path: str, task_id: str): 
    
    # Create necessary detection objects
    court_detector: CourtDetector = CourtDetector(output_resolution=(1920, 1080))
    camera_estimator: CameraEstimator = CameraEstimator(output_resolution=(1920, 1080))
    
    global object_detector
    if object_detector is None:
        object_detector = YoloDetector() 
    
    team_classifier: ColorHistogramClassifier = ColorHistogramClassifier(num_of_teams=3)
        
    team_detector: TeamDetector = TeamDetector(None, None)
    
    team_classifier.initialize_using_video(
        input_video_path, object_detector, 
        court_detector, training_frames=50)
    
    team_detector.player_clustering = team_classifier
    
    # Create an instance of video handler
    video_handler: VideoHandler = VideoHandler(input_video_path)

    analysis: GameAnalytics = GameAnalytics(video_handler.video_fps, task_id)
    analysis.infer_team_sides(
        input_video_path, court_detector, 
        object_detector, team_classifier, 
        camera_estimator, training_frames=1
    )

    
    # frame iterator to loop over input video frames
    frame_iterator: Generator[np.ndarray, None, None] = iter(video_handler.generate_frames())

    # output video path 
    output_video_path: str = util.create_output_dir(task_id) 
    
    video_writer: cv2.VideoWriter = video_handler.get_video_writer(f'{output_video_path}/detection.webm', (1920, 1080)) 
    map2d = video_handler.get_video_writer(f'{output_video_path}/map.webm', (920, 720))

    frame_index: int = 0
    for frame in frame_iterator:
        print(f'Processing {frame_index} - frame')
        if frame_index >= 15:
            break

        drawn_frame: np.ndarray = frame.copy()
        masked_court_image, masked_edge_image = court_detector.get_masked_and_edge_court(drawn_frame)

        player_boxes, ball_boxes = object_detector.detect(masked_court_image, team_classifier)

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
        top_view_frame = analysis.get_analytics()
        
        video_writer.write(output_frame)
        map2d.write(top_view_frame)
        frame_index += 1
    
    analysis.save_coords_data(analysis.player_list,
                              f'{output_video_path}/players.csv')
    
    # db.save_list_to_sql(analysis.player_list, engine)

    # dispose off resources
    video_writer.release()
    map2d.release()

    return True


if __name__ == '__main__':
    from inference.firebase import firestore
    task_id: str = 'manual-run'
    detect(input_video_path='./videos/fifa.mp4', task_id=task_id)
    detection_url = firestore.upload_file_to_firebase(f'out/manual-run/detection.webm', 'detection.webm', task_id)
    map_url = firestore.upload_file_to_firebase(f'out/manual-run/map.webm', 'map.webm', task_id)
    print(
        map_url, detection_url,
        end='\n'
    )