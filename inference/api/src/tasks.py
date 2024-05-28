import traceback
import inference.persistance.persist as db
from celery import shared_task
from inference.main import detect
from inference.firebase import firestore
from inference.util import utils
from celery import states
from celery.exceptions import Ignore
from inference.eventdetection import infer

# -----------------------------------------------------------
# The celery task for handling player detection and mapping
# the results to a 2d match + annotating the video. Then,
# it uploads the annotated videos to firebase, processes 
# output results and produces insights and saves that to a 
# database on gcp. That data is consumed in a bi dashboard.
# -----------------------------------------------------------
@shared_task(ignore_result=False)
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

            # save the list to sql
            # db.save_list_to_sql(player_list, engine)

            # dispose the engine
            # engine.dispose()

            return {
                'detection_vid': detection_vid_url,
                'map_vid': map_vid_url,
                'analysis': data
            }

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

        return video_url

    except Exception as e:
        print(traceback.format_exc())
        event_detection.update_state(state=states.FAILURE)
        raise Ignore()