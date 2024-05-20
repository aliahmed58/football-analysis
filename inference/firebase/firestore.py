import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from typing import List

# Use the application default credentials.
cred = credentials.ApplicationDefault()

firebase_admin.initialize_app(
    cred, {
        'storageBucket': 'fyp-testing-aa31e.appspot.com'
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

# Tests
if __name__ == '__main__':
    out = upload_file_to_firebase('../out/manual-run/detection.mp4', 'detection.mp4', 'out')
    print(out)