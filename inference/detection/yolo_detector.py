import torch
import inference.cvdetection as cvdetection
from typing import List
from inference.detection.teamdetector.BoundingBox import PlayerBoundingBox
import inference.util.utils as util

class YoloDetector:

    def __init__(self) -> None:
        torch.multiprocessing.set_start_method('spawn')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                    f'{util.get_project_root()}/weights/best.pt', device=0, 
                                    force_reload=True)

    def detect(self, frame, team_classifier):
        # do detections
        results = self.model(frame, size=1280)
        
        # get detections from the predicted results
        detections = cvdetection.Detection.get_detections_from_results(
            predictions=results.pred[0].cpu().numpy(),
            names=self.model.names
        ) 

        # filter detections by class names
        ball_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="ball")
        referee_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="referee")
        goalkeeper_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="goalkeeper")
        player_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="player")

        person_detections: List[cvdetection.Detection] = referee_detections + goalkeeper_detections + player_detections


        # ball and player boxes of type BoundingBox that are supported with the existing code we have
        ball_boxes: List[PlayerBoundingBox] = util.detections_to_bounding_box(frame, ball_detections)
        person_boxes: List[PlayerBoundingBox] = util.detections_to_bounding_box(frame, person_detections)


        if (len(person_boxes) > 0):
            person_boxes = team_classifier.infer(person_boxes)

        return person_boxes, ball_boxes
    
    def detect_objects(self, frame):
        # do detections
        results = self.model(frame, size=1280)
        
        # get detections from the predicted results
        detections = cvdetection.Detection.get_detections_from_results(
            predictions=results.pred[0].cpu().numpy(),
            names=self.model.names
        ) 

        # filter detections by class names
        ball_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="ball")
        referee_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="referee")
        goalkeeper_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="goalkeeper")
        player_detections: List[cvdetection.Detection] = cvdetection.filter_detections_by_class(detections=detections, class_name="player")

        person_detections: List[cvdetection.Detection] = referee_detections + goalkeeper_detections + player_detections


        # ball and player boxes of type BoundingBox that are supported with the existing code we have
        ball_boxes: List[PlayerBoundingBox] = util.detections_to_bounding_box(frame, ball_detections)
        person_boxes: List[PlayerBoundingBox] = util.detections_to_bounding_box(frame, person_detections)

        return person_boxes, ball_boxes