import numpy as np
import video
from typing import Dict, List
from util.containers import Rect, Color
from dataclasses import dataclass

# ----------------------------------------------------
# Basic yolo detection module for players, balls etc
# Methods related to detections, such as separating
# classification, generating reults and so on.
# ----------------------------------------------------

@dataclass
class Detection:
    rect: Rect
    class_id: int
    class_name: int
    confidence: float

    # ----------------------------------------------------
    # Takes in predictions from the yolo model inference
    # and outputs results as objects of Detection class
    # where each has a rect, class name and confidence etc.
    # ---------------------------------------------------- 
    @classmethod
    def get_detections_from_results(self, predictions: np.ndarray, names: Dict[int, str]) -> List:
        results = []
        for x_min, y_min, x_max, y_max, confidence, class_id in predictions:
            class_id=int(class_id)
            results.append(Detection(
                rect=Rect(
                    x=float(x_min),
                    y=float(y_min),
                    width=float(x_max - x_min),
                    height=float(y_max - y_min)
                ),
                class_id=class_id,
                class_name=names[class_id],
                confidence=float(confidence)
            ))
        return results
    
# -------------------------------------------------------
# filter detected objects given a class name i.e. ball 
# -------------------------------------------------------
def filter_detections_by_class(detections: List[Detection], class_name: str) -> List[Detection]:
    return [
        detection
        for detection
        in detections
        if detection.class_name == class_name
    ]

# ---------------------------------------------------------
# draw rects on detected objects on the frame to output a video
# ---------------------------------------------------------
def annotate(frame: np.ndarray, detections: List[Detection]) -> None:
    for detection in detections:
        frame = video.draw_rect(frame, detection.rect, Color.from_hex_string('FFFF00'), 2)