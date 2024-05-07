import cv2
from util import utils

def get_frames_from_video(video_file, num_of_frames):
    video_handler = VideoHandler(file=video_file, output_resolution=(1920, 1080))
    image_data = [video_handler.get_frame() for i in range(num_of_frames)]
    video_handler.release()
    return image_data

# import the necessary packages
import datetime

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (datetime.datetime.now() - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


class VideoHandler:

    def __init__(self, file, output_resolution=None):
        self.video_stream = cv2.VideoCapture(file)
        self.video_resolution = (
                int(self.video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        self.timestamp = 0
        self.frame_count = 0
        self.fps = FPS().start()

        if output_resolution is None:
            self.output_resolution = self.video_resolution
        else:
            self.output_resolution = output_resolution

        _, self.current_frame = self.video_stream.read()
        self.next_frame = None
        if self.current_frame is not None:
            _, self.next_frame = self.video_stream.read()
        else:
            print('No video file.')

        self.codec = None
        self.output_video_stream = None
        self.play_video = True

    def start_recording(self, output_file, recording_resolution):
        self.codec = cv2.VideoWriter_fourcc(*'XVID')

        output_fps = self.video_stream.get(cv2.CAP_PROP_FPS)
        self.output_video_stream = cv2.VideoWriter(output_file, self.codec, output_fps, recording_resolution)

    def has_frames(self):
        return self.play_video and self.next_frame is not None

    def get_frame(self):
        self.current_frame = self.next_frame
        _, self.next_frame = self.video_stream.read()
        self.fps.update()
        self.frame_count += 1
        self.timestamp = self.timestamp + 1000 / self.fps.fps()
        return cv2.resize(self.current_frame, self.output_resolution)

    def _record(self, frame):
        self.output_video_stream.write(frame)

    def release(self):
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
        self.video_stream.release()
        if self.output_video_stream is not None:
            self.output_video_stream.release()

    def show_image(self, window_title, frame, show_out=True):
        if self.output_video_stream is not None:
            self._record(frame)
        # change method to only show output if a boolean variable passed is true
        if show_out:
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.play_video = False
            if key == ord('p'):
                cv2.waitKey(-1)
