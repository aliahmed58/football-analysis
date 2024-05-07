import cv2
import numpy as np

class TeamDetector:
    def __init__(self, object_detector, player_clustering):
        self.object_detector = object_detector
        self.player_clustering = player_clustering

    @staticmethod
    def get_ball_color_label():
        return 255, 255, 255

    @staticmethod
    def draw_boxes(frame, boxes, line_width=2):
        image = np.copy(frame)

        for b in boxes:
            image = b.draw_box(image, line_width=line_width)
        return image

    @staticmethod
    def draw_on_field_points(frame, boxes):
        image = np.zeros_like(frame)
        
        for b in boxes:
            cv2.circle(image, b.get_field_point(), 3, b.color, 5)
        return image
