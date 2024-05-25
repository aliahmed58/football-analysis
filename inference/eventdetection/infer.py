import torch
import os
import cv2
import pandas as pd
import gc
from inference.eventdetection.efficientnet import EfficientNetB0ForCustomTask
from inference.util import utils
from torchvision.transforms import transforms
from PIL import Image

df = pd.DataFrame(columns=['Frame No','timestamp','Classifier1','Classifier2'])
class_to_idx_mapping = {'Center': 0, 'Left': 1, 'Right': 2,'Free-Kick':3,'Penalty': 4, 'Tackle': 5,'To Substitue' : 6,'Cards':7,'Corner': 8}
class_to_idx_mapping_class1 = {'Event': 0, 'Soccer': 1}

path = f'{utils.get_project_root()}/eventdetection'
weights_path = f'{path}/weights'
out_path = f'{utils.get_project_root()}/out'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def map_class_index(output_index, class_to_idx_mapping):
    for class_name, idx in class_to_idx_mapping.items():
        if idx == output_index:
            return class_name
    return None

def local_classifier1(num_classes=2):
    model = EfficientNetB0ForCustomTask(num_classes).to(device)
    model.load_state_dict(torch.load(f'{weights_path}/layer1.pth',map_location=device))
    model.eval()

    return model

def local_classifier2(num_classes=9):
    model = EfficientNetB0ForCustomTask(num_classes).to(device)
    model.load_state_dict(torch.load(f'{weights_path}/layer2.pth',map_location=device))
    model.eval()

    return model

# global variables acting as a singleton
classifier1_model = None
classifier2_model = None

def video_classifier(video_path: str, task_id: str):
    
    # Get the video
    video = cv2.VideoCapture(video_path)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # Get video properties
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frameduration = 1/fps
    framecount = 0
    
    # Initialize classifiers
    global classifier1_model, classifier2_model

    if classifier1_model is None:
        classifier1_model = local_classifier1()
    
    if classifier2_model is None:
        classifier2_model = local_classifier2()

    # Transformations definition
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Define output video writer
    output_path = f'{out_path}/{task_id}'
    # create path if not exists
    os.makedirs(output_path)
    vid_out_path = f'{out_path}/{task_id}/events.webm'
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    output_video = cv2.VideoWriter(vid_out_path, fourcc, fps, (width, height))

    # Read the video and classify each frame
    for i in range(int(frames)):
        ret, frame = video.read()
        if not ret:
            break
        else:
            print(f'Processing frame #{i}')
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = transform(image).unsqueeze(0).to(device)
            timestamp = framecount * frameduration
            framecount += 1
            with torch.no_grad():
                output1 = classifier1_model(image_tensor)
                _, predicted1 = torch.max(output1, 1)
                if predicted1.item() == 1:
                    continue
                else:
                    _, predicted2 = torch.max(classifier2_model(image_tensor), 1)
                    # Draw classifier outputs on the frame
                    predicted_class1 = map_class_index(predicted1.item(),class_to_idx_mapping_class1)
                    predicted_class2 = map_class_index(predicted2.item(), class_to_idx_mapping)
                    cv2.putText(frame, f"Classifier1: {predicted_class1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Classifier2: {predicted_class2}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Write frame with text to the output video
                    output_video.write(frame)
                    # Show frame in a window
                    # cv2.imshow('Frame', frame)
                    # Store results in DataFrame
                    df.loc[i] = [i, timestamp, predicted1.item(), predicted2.item()]
                    # Wait for key press
                    # cv2.waitKey(1)
    
    # Close video window
    # cv2.destroyAllWindows()

    torch.cuda.empty_cache()

    # Save results to CSV
    df.to_csv(f'{out_path}/{task_id}/events.csv', index=False)

    # Release output video writer
    output_video.release()
