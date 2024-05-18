import os
from . import config
import numpy as np
import scipy.io as sio
from PIL import Image
from typing import List
from pathlib import Path
from inference.cvdetection import Detection
from inference.detection.teamdetector.BoundingBox import PlayerBoundingBox

# ----------------------------------------------
# utility methods reused throughout the project
# ----------------------------------------------


# get the root of project
def get_project_root() -> str:
    return f'{Path(__file__).parent.parent}'

# path to our input videos
def get_input_videos_path() -> str:
    return f'{get_project_root()}/videos'

# path to yolo trained weights and other model weights
def get_weights_path() -> str:
    return f'{get_project_root()}/weights'

# path to two pix2pix model
def get_two_pix2pix_model_path() -> str:
    return f'{get_project_root()}/mlmodels/generated_models/two_pix2pix/'


# creates the output directory and returns the path as string
def create_output_dir() -> str:
    no_of_dirs = len(os.listdir(f'{get_project_root()}/{config.OUTPUT_DIR_NAME}/'))
    mkdirpath = f'{get_project_root()}/{config.OUTPUT_DIR_NAME}/run-{no_of_dirs + 1}'
    if os.path.exists(mkdirpath):
        return
    os.makedirs(mkdirpath)
    
    return mkdirpath

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

# save an np ndarray as image given the path
def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

# concatenate multiple images
def concatenate_multiple_images(*images):
    img_height = images[0].shape[0]
    img_delimiter = np.ones((img_height, 2, 3), dtype=np.uint8) * 128
    image_to_be_saved = img_delimiter
    for im in images:
        image_to_be_saved = np.concatenate((image_to_be_saved, im, img_delimiter), axis=1)
    return image_to_be_saved

def get_camera_estimator_files_path():
    return f'{get_project_root()}/detection/cameraestimator/files'

# get binary court using the mat lab file
def get_binary_court():
    return sio.loadmat(f'{get_camera_estimator_files_path()}/worldcup2014.mat')

# path to the siamese.pth model file
def get_siamese_model_path():
    return f'{get_project_root()}/mlmodels/generated_models/siamese'

# Detection instacne to BoundingBox instance convertor 
# easier than to refactor a lot of already existing code.
def detections_to_bounding_box(image: np.ndarray, dbox: List[Detection]) -> List[PlayerBoundingBox]:
    boxes: List[PlayerBoundingBox] = []
    for b in dbox:
        x1, y1 = b.rect.top_left.x, b.rect.top_left.y
        x2, y2 = b.rect.bottom_right.x, b.rect.bottom_right.y
        boxes.append(PlayerBoundingBox(image, box_coords=(x1, y1, x2, y2), score=b.confidence))
    
    return boxes