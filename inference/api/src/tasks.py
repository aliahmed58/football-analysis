import traceback
import inference.persistance.persist as db
import numpy as np
import json
from celery import shared_task
from inference.main import detect
from inference.firebase import firestore
from inference.util import utils
from celery import states
from celery.exceptions import Ignore
from inference.eventdetection import infer
from inference.analysis import passing, possesion, pressure, receiving
from flask import jsonify, make_response

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

@shared_task(ignore_result=False, serializer='json')
def test_insights():
    task_id = 'manual-run'
    input_file = f'{utils.get_project_root()}/out/{task_id}/players.csv'

    df_passing = passing.calc_passing(input_file)
    df_receiving = receiving.calc_receiving(input_file)
    df_pressure = pressure.calc_pressure(input_file)
    df_possession = possesion.calc_possession(input_file)

    # Home team
    sides = ['Home', 'Away']
    # sample output json object, base dict that needs to be passed should be like this.
    data: dict = {
        'Home': {
            'images': {

            },
            'passes': {

            }
        },
        'Away': {
            'images': {

            },
            'passes': {

            }
        }
    }

    for side in sides:
        # passing
        passing.create_pass_map_complete(df_passing, side, task_id, data)
        passing.create_pass_map_incomplete(df_passing, side, task_id, data)

        # receiving
        receiving.create_receiving_map(df_receiving, side, task_id, data)

        # pressure
        pressure.create_pressure_map(df_pressure, side, task_id, data)

        # possession
        possesion.create_heatmap(df_possession, side, task_id, data)

    return data


# -----------------------------------------------------------
# The celery task for handling player detection and mapping
# the results to a 2d match + annotating the video. Then,
# it uploads the annotated videos to firebase, processes 
# output results and produces insights and saves that to a 
# database on gcp. That data is consumed in a bi dashboard.
# -----------------------------------------------------------
@shared_task(ignore_result=False, serializer='json')
def infer_footage(full_video_name):
    task_id = infer_footage.request.id
    input_video = None

    # try downloading the uploaded video and see if it exists
    try:
        input_video = firestore.download_file(f'upload/{full_video_name}', 'videos')
    except Exception as e:
        print(traceback.format_exc())
        print('Could not find file on cloud, are you sure ID was correct.')
        infer_footage.update_state(state=states.FAILURE)
        raise Ignore()

    try:
        # start the inference method that does detection and extracts data
        data = detect(
        input_video_path=input_video,
        task_id=task_id
        )
        # if it ran successfully, upload the output videos to firebase cloud storage
        if data:

            # engine = db.get_engine()

            # saved as webm so it can be easily 
            detection_vid_url = firestore.upload_file_to_firebase(
                f'{utils.get_project_root()}/out/{task_id}/detection.webm',
                'detect/detection.webm',
                task_id
            )
            map_vid_url: str = firestore.upload_file_to_firebase(
                f'{utils.get_project_root()}/out/{task_id}/map.webm',
                'detect/map.webm',
                task_id
            )

            data['detection_vid'] = detection_vid_url
            data['map_vid'] = map_vid_url

            return data

    except Exception as e:
        print('Exception occured ', e)
        print(traceback.format_exc())
        infer_footage.update_state(state=states.FAILURE)
        raise Ignore()
    
# -----------------------------------------------------------
# The celery task for event detection, which runs on a video
# and produces an annotated video showing which football
# event is occuring on the current frame. This data is also
# written to a csv file which will be uploaded to sql and 
# the annotated video is uploaded to firebase to be accessed
# by a react frontend. The value returned is the link after
# task is completed successfuly.
# -----------------------------------------------------------
@shared_task(ignore_result=False)
def event_detection(full_video_name):
    task_id = event_detection.request.id
    input_video = None
    
    # try downloading the uploaded video and see if it exists
    try:
        input_video = firestore.download_file(f'upload/{full_video_name}', 'videos')
    except Exception as e:
        print(traceback.format_exc())
        print('Could not find file on cloud, are you sure ID was correct.')
        infer_footage.update_state(state=states.FAILURE)
        raise Ignore()
    
    try:
        infer.video_classifier(input_video, task_id)
        
        video_url = firestore.upload_file_to_firebase(
            f'{utils.get_project_root()}/out/{task_id}/events.webm',
            'event/events.webm',
            task_id
        )

        csv_url = firestore.upload_file_to_firebase(
            f'{utils.get_project_root()}/out/{task_id}/events.csv',
            'event/events.csv',
            task_id
        )

        return {
            'event_vid': video_url,
            'csv': csv_url
        }

    except Exception as e:
        print(traceback.format_exc())
        event_detection.update_state(state=states.FAILURE)
        raise Ignore()