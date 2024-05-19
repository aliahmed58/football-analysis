from celery import shared_task

from inference.main import detect

@shared_task(ignore_result=False)
def infer_footage():
    task_id = infer_footage.request.id
    detect(
        input_video_path='../videos/fifa.mp4',
        task_id=task_id
    )