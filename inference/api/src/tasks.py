import traceback
from celery import shared_task
from inference.main import detect
from inference.firebase import firestore
from inference.util import utils

@shared_task(ignore_result=False)
def infer_footage():
    task_id = infer_footage.request.id
    try:
        # start the inference method that does detection and extracts data
        success: bool = detect(
        input_video_path='../videos/fifa.mp4',
        task_id=task_id
        )

        # if it ran successfully, upload the output videos to firebase cloud storage
        if success:
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

            return [detection_vid_url, map_vid_url]
        return False

    except Exception as e:
        print('Exception occured ', e)
        print(traceback.format_exc())
        return False