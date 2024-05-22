from flask import Blueprint, request
from celery.result import AsyncResult

bp = Blueprint('detection', __name__, url_prefix='/')

from inference.api.src import tasks

@bp.route('/hello', methods=['GET'])
def hello_world():
    return "<h1> Hello world </h1>"

@bp.route('/infer/<video_id>', methods=['GET'])
def detect(video_id: str):
    results = tasks.infer_footage.delay(video_id)
    return {'result_id': results.id}

@bp.get("/result/<id>")
def task_result(id: str) -> dict[str, object]:
    result = AsyncResult(id)
    return {
        "ready": result.ready(),
        "successful": result.successful(),
        "value": result.result if result.ready() else False,
        "state": result.state
    }