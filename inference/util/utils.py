import os
import numpy as np
from PIL import Image
from pathlib import Path

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


def create_dir_if_not_exists(dirpath: str) -> None:
    if os.path.exists(dirpath):
        return
    os.makedirs(dirpath)

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

