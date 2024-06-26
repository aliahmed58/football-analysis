from flask import Blueprint, request
from celery.result import AsyncResult
from inference.api.src import tasks
from flask import jsonify, Response
from inference.util import utils
import os
import json
import numpy as np

bp = Blueprint('detection', __name__, url_prefix='/infer')
out_dir = f'{utils.get_project_root()}/out'

def check_if_done(video_name: str):
    path = f'{utils.get_project_root()}/videos/{video_name}'
    return os.path.isfile(path)

@bp.route('/hello', methods=['GET'])
def hello_world():
    results = tasks.test_insights.delay()
    return {'result_id': results.id}

@bp.get('/hello/result/<id>')
def resulttest(id: str):
    result = AsyncResult(id)
    return {
        'value': result.result if result.ready() else False
    }

# ------------------------------------------------
# Routes for detection and 2d mapping
# ------------------------------------------------

@bp.route('/detect/<video_id>', methods=['GET'])
def detect(video_id: str):
    if check_if_done(f'detect/{video_id}'):
        return {
            'exists': True
        }
    
    if f'detect/{video_id}' in tasks.current:
        return {
            'pending': True
        }
    tasks.current.add(f'detect/{video_id}')
    results = tasks.infer_footage.delay(video_id)
    return {'result_id': results.id}

@bp.get("/detect/result/<id>")
def task_result(id: str) -> dict[str, object]:
    if os.path.exists(f'{out_dir}/{id}'):
        if os.path.isfile(f'{out_dir}/{id}/res.json'):
            with open(f'{out_dir}/{id}/res.json') as f:
                data = json.load(f)
                return data

    result = AsyncResult(id)
    res = {
        "ready": result.ready(),
        "successful": result.successful(),
        "value": result.result if result.ready() else False,
        "state": result.state
    }

    if result.ready():
        with open(f'{out_dir}/{id}/res.json', 'w') as f:
            json.dump(res, f)

    return res

# ------------------------------------------------
# Routes for event detection
# ------------------------------------------------
@bp.route('/events/<video_id>', methods=['GET'])
def event_detect(video_id: str):
    if check_if_done(f'event/{video_id}'):
        print(True)
        return {
            'exists': True
        }
    if f'event/{video_id}' in tasks.current:
        print(True)
        return {
            'pending': True
        }
    tasks.current.add(f'event/{video_id}')
    results = tasks.event_detection.delay(video_id)
    return {
        'result_id': results.id
    }

@bp.get('/events/result/<id>')
def event_result(id: str) -> dict[str, object]:
    if os.path.exists(f'{out_dir}/{id}'):
        if os.path.isfile(f'{out_dir}/{id}/res.json'):
            with open(f'{out_dir}/{id}/res.json') as f:
                data = json.load(f)
                return data

    result = AsyncResult(id)
    res = {
        "ready": result.ready(),
        "successful": result.successful(),
        "value": result.result if result.ready() else False,
        "state": result.state
    }

    if result.ready():
        with open(f'{out_dir}/{id}/res.json', 'w') as f:
            json.dump(res, f)

    return res