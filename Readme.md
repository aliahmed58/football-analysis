# Inference module for football detection

## inference

Football video detection through yolov5 and mapping player coordinates on 2D Map then extracting that data in a tabular form.

Runs from `main.py`

## API

Flask api with Celery and eventlet to handle background processing. The flask server is responsible for handling api endpoints listed below and the tasks received are handled by celery. Every task has a unique id that is also used as the output directory name, and the unique id for that specific run. Tasks are queued in a redis broker and are executed one by one. Parallel execution is avoided due to exceptions and complications with CUDA.

List of endpoints:

 `host:port/infer/:<video_id>` 

_Request_: Sends a task to the server, video with the given unique video filename is fetched from firebase cloud storage under `uploads/` folder.

_Response_: Sends back a result_id, that is the task unique id assigned to it

- `host:port/result/:<id>`

_Request_: Request the task status from the server providing the result_id given in `/infer` request.

_Response_: Return a json object with the following fields:
- `ready: Boolean` - tells whether a task is completed or not
- `successfull: Boolean` - tells whether the task completed successfully or failed
- `value: Any` - Returns the list of public links annotated and 2d map video uploaded to firebase or False if task failed.
- `status: 'FAILURE' or 'SUCCESS'` - Returns FAILURE if task is failed due to any error.


# Setup

Poetry is used to manage dependencies in this project and thus highly recommended to use it for installation.

Python version: 3.9.18 -- TODO: make it flexible to be any of 3.9

CUDA compatible GPU with cuda toolkit installed is needed.

1. Install poetry by `pip install poetry`

2. --Optional-- Set poetry to make venv in the project directory by `poetry config virtualenvs-in.project true`

3. Install dependencies using `poetry install` in the root directory.

4. Make sure the package is installed by activating the venv and running `pip list`. The package `inference` should show up there.

5. Download files DualPix2Pix models and weights for detection 
      - segmentation model weights: https://drive.google.com/file/d/1QCinahFH_830nH2RqwgoT8jehqxJgHQK/view?usp=share_link
      - line detection model weights: https://drive.google.com/file/d/1QzJzSUP9Zmqc4Eiko3dS1ZZTDmSQ0E10/view?usp=share_link
      - best.pt - yolov5 trained weights for football: https://drive.google.com/file/d/1DbtscjXnWmimBV-4cqdMKnKJWhmv8NSg/view?usp=drive_link

6. Place the two `.pth` files in directory `inference/mlmodels/generated_models/two_pix2pix/`.

7. Place the `best.pt` file in a new directory named weights in `inference`.

8. Create a `dbkeys.py` file in `inference/persistance` to save Database configurations with the following names:
      - `DB_USER = '<user>'`
      - `DB_PASS = '<pass>'`
      - `DB_HOST = '<host>'`
      - `DB_NAME = '<db name>'`
      - `CONNECTION_NAME = '<connection name>'`

9. Connect with firebase project and enable cloud admin API from google console.

# Run

## Inference Only

To run only the inference module, run the `main.py` file by `python main.py`, make sure the venv is activated.

In the main file provide the video file path and task id to the `detect` function call.

## Inference with web server (Flask + Celery)

Make sure redis is installed and running since Celery makes use of reddit. Celery is not supported on windows with the version in this project.

To run it as a server that handles tasks on API endpoints listed above, do the following:

1. Open a terminal in `inference/api` and run celery by the command:
```
celery -A make_celery worker --loglevel INFO -P solo
```

2. Open another terminal also in `inference/api` and run flask server by the command:
```
flask -A src run --debug
```

The above api endpoints can be tested then.