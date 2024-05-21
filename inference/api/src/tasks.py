import traceback
import inference.persistance.persist as db
from celery import shared_task
from inference.main import detect
from inference.firebase import firestore
from inference.util import utils

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

    try:
        # start the inference method that does detection and extracts data
        player_list = detect(
        input_video_path=input_video,
        task_id=task_id
        )
        # if it ran successfully, upload the output videos to firebase cloud storage
        if player_list:

            engine = db.get_engine()

            # saved as webm so it can be easily 
            detection_vid_url = firestore.upload_file_to_firebase(
                f'{utils.get_project_root()}/out/{task_id}/detection.webm',
                'detection.webm',
                task_id
            )
            map_vid_url: str = firestore.upload_file_to_firebase(
                f'{utils.get_project_root()}/out/{task_id}/map.webm',
                'map.webm',
                task_id
            )

            # save the list to sql
            db.save_list_to_sql(player_list, engine)

            # dispose the engine
            engine.dispose()

            return [detection_vid_url, map_vid_url]

    except Exception as e:
        print('Exception occured ', e)
        print(traceback.format_exc())