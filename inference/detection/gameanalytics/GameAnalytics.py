import cv2
import numpy as np
import datetime
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression
import queue
import pandas as pd

from detection.cameraestimator.CameraEstimator import estimate
from .TopView import TopViewer


from util.video_handling import get_frames_from_video
from util import utils


class GameAnalytics:
    RIGHT = 'right'
    LEFT = 'left'
    color_names = {
        (255, 255, 255): 'White',
        (155, 155, 155): 'No',
        (255, 0, 0): 'Blue',
        (0, 0, 255): 'Red',
        (0, 255, 0): 'Green',
        (0, 255, 255): 'Yellow',
        (255, 0, 255): 'Purple'
    }

    def __init__(self, vid_fps: int):
        self.fps = vid_fps
        self.player_list = []
        self.ball_list = []
        self.pass_list = []
        self.top_viewer = TopViewer(court_dimensions=(920, 592))
        self.court_dimensions = self.top_viewer.court_dimensions
        court_w = 106
        court_h = 69
        desired_w, desired_h = self.court_dimensions
        self.possession_radius = ((desired_w / court_w) + (desired_h / court_h)) / 2 + 10
        self.possession_last_point = None
        self.default_color = (155, 155, 155)
        self.ball_point = None
        self.color_team_in_possession = self.default_color
        self.semantics = '-'
        self.current_points = None
        self.history_size = 100
        self.ball_history = queue.deque(maxlen=self.history_size)
        self.frame = 0
        self.start_time = datetime.datetime.now()

    # @utils.logging_time
    def update(self, homography, boxes):
        self.current_points = self.top_viewer.convert_boxes_to_topview_points(homography, boxes)

        ball_points = filter(lambda x: x.color == (255, 255, 255), self.current_points)
        self.ball_history.extend([p.coords for p in ball_points])
        
        # TODO Get different goal keepers as well and keep colors dynamic
        player_points = filter(lambda x: x.color != (255, 255, 255), self.current_points)
        self._find_ball_possession()

        # keep appending the x, y coordinates with frame number to a list so it can be saved as csv later 
        # list -> dataframe -> csv
        self._append_rows_to_list(player_points, self.player_list)

        self.frame += 1

    # @utils.logging_time
    def get_analytics(self):
        top_view_frame = self.get_top_view()
        info_board = self._get_analytics_board()
        
        return np.concatenate([top_view_frame, info_board], axis = 0)

    def get_top_view(self):
        return self.top_viewer.project_on_topview(self.current_points, self.ball_history)

    def get_info_board(self):
        self._find_ball_possession()
        self._find_semantics()
        return self._get_analytics_board()

    def save_coords_data(self, out_list: list, out_path: str) -> None:
        df = pd.DataFrame(out_list)
        df.to_csv(out_path, index=False)

    def _find_ball_possession(self):
        ball_points = self._get_current_ball_point()
        player_points = self._get_current_player_points()
        self.color_team_in_possession = self.default_color

        if len(ball_points) == 0:
            return

        self.ball_point = np.array(ball_points[0].coords)

        for b in ball_points:
            b_point = np.array(b.coords)
            for p in player_points:
                player_point = np.array(p.coords)
                dist = norm(b_point - player_point)
                if dist < self.possession_radius + 10:
                    self.color_team_in_possession = p.color
                    p.possession = 1
                    break

    def _find_semantics(self):
        self.semantics = '-'

        if self.ball_point is None:
            return
        side = self._ball_court_location()

        if self.color_team_in_possession == self.top_viewer.left_side_color and side == self.LEFT:
            self.semantics = 'defending'
        elif self.color_team_in_possession == self.top_viewer.left_side_color and side == self.RIGHT:
            self.semantics = 'attacking'
        elif self.color_team_in_possession == self.top_viewer.right_side_color and side == self.LEFT:
            self.semantics = 'attacking'
        elif self.color_team_in_possession == self.top_viewer.right_side_color and side == self.RIGHT:
            self.semantics = 'defending'

    def _get_analytics_board(self):
        desired_w, _ = self.court_dimensions
        board = np.zeros((128, desired_w, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_width = 30
        team_in_ball_possession = self.color_names[self.color_team_in_possession]

        i = 25
        cv2.putText(board, f'{team_in_ball_possession} team has ball possession',
                    (1, i), font, 1, self.color_team_in_possession, 2, cv2.LINE_AA)

        i += line_width
        if self.semantics == '-':
            cv2.putText(board, f'Semantics: {self.semantics}', (1, i), font, 1,
                        self.default_color, 2, cv2.LINE_AA)
        else:
            cv2.putText(board, f'Semantics: {team_in_ball_possession} team is {self.semantics}', (1, i), font, 1,
                        self.default_color, 2, cv2.LINE_AA)
        return board

    def _get_current_ball_point(self):
        return list(filter(lambda x: x.color == (255, 255, 255), self.current_points))

    def _get_current_player_points(self):
        return list(filter(lambda x: x.color != (255, 255, 255), self.current_points))

    def _ball_court_location(self):
        desired_w, _ = self.court_dimensions
        x, y = self.ball_point
        side = self.RIGHT
        if x < desired_w / 2:
            side = self.LEFT
        return side

    def infer_team_sides(self, video_file, court_detector, object_detector, team_classifier, camera_estimator, training_frames):
        
        image_data = get_frames_from_video(video_file, training_frames)
        topview_points = self._extract_training_samples(
            image_data, team_classifier, object_detector, court_detector, camera_estimator)
        
        team_colors = team_classifier.colors
        groups = self._get_all_team_groups(topview_points, team_colors)
        groups, soccer_team_a, soccer_team_b = self._get_soccer_team_groups(groups)

        soccer_data = soccer_team_a + soccer_team_b
        x_data, y_data = self._prepare_for_training(soccer_data)

        left_side_color, right_side_color = self._find_team_side_colors(x_data, y_data, groups)
        self.top_viewer.set_team_side_colors(left_side_color, right_side_color)

    def _extract_training_samples(self, image_data, team_classifier, object_detector, court_detector, camera_estimator):
        output_samples = []
        for frame in image_data:
            masked_court_image, masked_edge_image = court_detector.get_masked_and_edge_court(frame)

            players, ball = object_detector.detect(frame, team_classifier)
            boxes = players + ball
            _ = estimate(camera_estimator, masked_edge_image)
            points = self.top_viewer.convert_boxes_to_topview_points(camera_estimator.last_estimated_homography, boxes)
            output_samples.extend(points)

        return output_samples

    @staticmethod
    def _get_all_team_groups(points, colors):
        groups = []
        for color in colors:
            group = list(filter(lambda x: x.color == color, points))
            groups.append(group)
        return groups

    @staticmethod
    def _get_soccer_team_groups(groups):
        groups = sorted(groups, key=len, reverse=True)
        for i in range(2):
            for point in groups[i]:
                point.label = i
        return groups, groups[0], groups[1]

    @staticmethod
    def _prepare_for_training(data):
        x = np.zeros((len(data), 2))
        y = np.zeros((len(data), 1))
        for i in range(len(data)):
            x[i, :] = data[i].coords
            y[i] = data[i].label
        return x, y


    def _find_team_side_colors(self, x_data, y_data, groups):
        classifier = LogisticRegression(C=1e5)
        classifier.fit(x_data, y_data)
        left = int(classifier.predict(np.array([[0, 300]])))
        right = int(classifier.predict(np.array([[1000, 300]])))
        left_color = groups[left][0].color
        right_color = groups[right][0].color

        # image = plot.plot_logistic_regression_points(classifier, x_data, y_data)
        # court = cv2.resize(self.top_viewer.court, (1200,800))
        # output_frame = cv2.addWeighted(src1=court,
        #                                src2=image,
        #                                alpha=.4, beta=.9, gamma=0.)

        return left_color, right_color

    def _scale_coordinates(self, x, y):
        x_scaled = ((x) / (920)) * (120)
        y_scaled = ((y) / (592)) * (80)

        return int(x_scaled), int(y_scaled)
    

    def _append_rows_to_list(self, rows, data_list):
        for point in rows:
            dict1: dict = {}
            x_scaled, y_scaled = self._scale_coordinates(point.coords[0], point.coords[1])
            t = datetime.datetime.now() - self.start_time
            actual_time = t.seconds / self.fps
            dict1.update({
                'teamId': self.color_names.get(point.color, "Undefined"),
                'frame': self.frame,
                'x': x_scaled,
                'y': y_scaled,
                'ball_posession': point.possession,
                'timestamp': actual_time
            })
            data_list.append(dict1)