import firebase_admin
import os
from firebase_admin import credentials
from firebase_admin import storage
from typing import List
from inference.util import utils
import traceback

# Use the application default credentials.
cred = credentials.Certificate(f'{utils.get_project_root()}/firebase/fanKey.json')

firebase_admin.initialize_app(
    cred, {
        'storageBucket': 'fyp-final-e36de.appspot.com'
    }
)

bucket = storage.bucket()

# Method to upload video to firebase 
# to be used in api/src/tasks.py after detection is completed
# only import once, or create a singleton
# Takes filepath and filename, returns the public cloud url for uploaded resource
def upload_file_to_firebase(file_path: str, filename: str, task_id: str) -> str:
    video_blob = bucket.blob(f'{task_id}/{filename}')
    video_blob.upload_from_filename(file_path)
    video_blob.make_public()
    print(f'Video successfully uploaded to firebase - id: {filename}')
    return video_blob.public_url

def get_file_url(filename: str):
    try:
        blob = bucket.blob(filename)
        return blob.public_url
    except Exception as e:
        print(e)
        print('blob does not exist')

def get_filename_from_blob(blob) -> str:
    filename = str(blob.name).split('/')
    filename = filename[len(filename) - 1]
    return filename

# returns downloaded video full path on success
def download_file(file_path: str, out_directory: str) -> str:

    out_path = f'{utils.get_project_root()}/{out_directory}'

    # make directory if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    try:
        blob = bucket.blob(file_path)
        filename: str = get_filename_from_blob(blob)
        blob.download_to_filename(f'{out_path}/{filename}')
        print(f'Video succesfully downloaded and saved to {out_path}')
        return f'{out_path}/{filename}'
    except Exception as e:
        print(traceback.format_exc())
        print('Error while downloading video')
    

# Tests
if __name__ == '__main__':
    # out = upload_file_to_firebase('../out/manual-run/detection.mp4', 'detection.mp4', 'out')

    # print(out)

    download_file('upload/0052c15f-9a06-4f8b-a029-3e58a8e1622efifa.mp4', 'videos')