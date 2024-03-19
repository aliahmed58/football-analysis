# Inference module for football detection

### Run

All commands from root of the project

- Download files DualPix2Pix models for court detection 
      - segmentation model weights: https://drive.google.com/file/d/1QCinahFH_830nH2RqwgoT8jehqxJgHQK/view?usp=share_link
      - line detection model weights: https://drive.google.com/file/d/1QzJzSUP9Zmqc4Eiko3dS1ZZTDmSQ0E10/view?usp=share_link
- Place downloaded files in directory _/models/generated_models/two_pix2pix/_
- Install CUDA toolkit 11.8.x from NVIDIA website
- Torch should be packaged with CUDA. ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117``` For more info check: [cuda site](https://pytorch.org/get-started/locally/)

1. Create a virtual environment in the root directory (preferred) using ```python3.8 -m venv ./venv```

2. Install requiremnts using ```pip install -r requirements.txt```

3. Install the project as package ```pip install -e .``` (dot at the end for current directory)

4. Create a ```inference/videos``` in inference folder and add your input video (mp4)

5. Set video name in ```util/config.py``` constants

6. Run main file from ```python3.8 main.py``` in ```/inference``` directory