from flask import Blueprint, request
from celery.result import AsyncResult
from inference.api.src import tasks

bp = Blueprint('detection', __name__, url_prefix='/infer')


@bp.route('/hello', methods=['GET'])
def hello_world():
    return "<h1> Hello world </h1>"

# ------------------------------------------------
# Routes for detection and 2d mapping
# ------------------------------------------------

@bp.route('/detect/<video_id>', methods=['GET'])
def detect(video_id: str):
    results = tasks.infer_footage.delay(video_id)
    return {'result_id': results.id}

@bp.get("/detect/result/<id>")
def task_result(id: str) -> dict[str, object]:
    result = AsyncResult(id)
    return {
        "ready": result.ready(),
        "successful": result.successful(),
        "value": result.result if result.ready() else False,
        "state": result.state
    }

# ------------------------------------------------
# Routes for event detection
# ------------------------------------------------
@bp.route('/events/<video_id>', methods=['GET'])
def event_detect(video_id: str):
    results = tasks.event_detection.delay(video_id)
    return {
        'result_id': results.id
    }

@bp.get('/events/result/<id>')
def event_result(id: str) -> dict[str, object]:
    result = AsyncResult(id)
    return {
        "ready": result.ready(),
        "successful": result.successful(),
        "value": result.result if result.ready() else False,
        "state": result.state
    }