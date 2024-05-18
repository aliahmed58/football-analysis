from celery import shared_task

from inference.main import detect

@shared_task(ignore_result=False)
def infer_footage():
    return 2 + 2