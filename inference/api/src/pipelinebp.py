from flask import Blueprint, redirect
from celery.result import AsyncResult

bp = Blueprint('detection', __name__, url_prefix='/')

from inference.api.src import tasks

@bp.route('/', methods=['GET'])
def detect():
    results = tasks.infer_footage.delay()
    return {'result_id': results.id}

@bp.get("/result/<id>")
def task_result(id: str) -> dict[str, object]:
    result = AsyncResult(id)
    return {
        "ready": result.ready(),
        "successful": result.successful(),
        "value": result.result if result.ready() else None,
    }