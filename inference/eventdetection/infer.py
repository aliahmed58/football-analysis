import torch
import os
from torchvision.transforms import transforms
from PIL import Image
import cv2
import pandas as pd
from inference.eventdetection.efficientnet import EfficientNetB0ForCustomTask
from inference.util import utils

df = pd.DataFrame(columns=['Frame No','timestamp','Classifier1','Classifier2','Cards'])
class_to_idx_mapping = {'Center': 0, 'Left': 1, 'Right': 2,'Free-Kick':3,'Penalty': 4, 'Tackle': 5,'To Substitue' : 6,'Cards':7,'Corner': 8}
class_to_idx_mapping_class1 = {'Event': 0, 'Soccer': 1}
class_to_idx_mapping_cards ={ 'Red Card': 0, 'Yellow Card': 1}
path = utils.get_project_root()
weights_path = f'{path}/eventdetection/weights'
out_path = f'{path}/out'
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


def local_classifier3(num_classes=2):
    model = EfficientNetB0ForCustomTask(num_classes).to(device)
    model.load_state_dict(torch.load(f'{weights_path}/cards.pth',map_location=device))
    model.eval()

    return model

classifier1_model, classifier2_model, cards_model = None, None, None

def video_classifier(video_path: str, task_id: str):

    
    event_detected = False
    event_start_frame = None
    event_start_timestamp = None
    consecutive_count = 0
    last_predicted_class2 = None
    events = []

    # Get the video
    video = cv2.VideoCapture(video_path)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frameduration = 1/fps
    framecount = 0

    global classifier1_model, classifier2_model, cards_model

    if classifier1_model is None:
        classifier1_model = local_classifier1()

    if classifier2_model is None:    
        classifier2_model = local_classifier2()
    
    if cards_model is None:
        cards_model = local_classifier3()

    # Transformations definition
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Define output video writer
    output_path = f'{out_path}/{task_id}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    output_video = cv2.VideoWriter(f'{output_path}/events.webm', fourcc, fps, (width, height))

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
                output2 = classifier2_model(image_tensor)
                _, predicted2 = torch.max(output2, 1)
                if predicted1.item() == 1:
        # If predicted1 is 1, write its value on the frame
                    predicted_class1 = map_class_index(predicted1.item(), class_to_idx_mapping_class1)
                    cv2.putText(frame, f"Classifier1: {predicted_class1}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.putText(frame, f"Classifier2: none", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.putText(frame, f"Classifier3: none", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    df.loc[i] = [i, timestamp, predicted_class1, None, None]
                else:
        # If predicted1 is not 1, use the second classifier
                    output2 = classifier2_model(image_tensor)
                    _, predicted2 = torch.max(output2, 1)


                if predicted2.item() == 7:
            # If predicted2 is 7, use the third classifier
                    output3 = cards_model(image_tensor)
                    _, predicted3 = torch.max(output3, 1)
                    predicted_class3 = map_class_index(predicted3.item(), class_to_idx_mapping_cards)
                    cv2.putText(frame, f"Classifier3: {predicted_class3}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Classifier1: {predicted_class1}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Classifier2: {predicted_class2}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    df.loc[i] = [i, timestamp, predicted_class1, predicted_class2, predicted_class3]
                else:
            # If predicted2 is not 7, write its value on the frame
                    predicted_class1 = map_class_index(predicted1.item(), class_to_idx_mapping_class1)
                    cv2.putText(frame, f"Classifier1: {predicted_class1}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    predicted_class2 = map_class_index(predicted2.item(), class_to_idx_mapping)
                    cv2.putText(frame, f"Classifier2: {predicted_class2}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    df.loc[i] = [i, timestamp, predicted_class1, predicted_class2, None]

    # Write frame with text to the output video
                output_video.write(frame)    
                
    # Save results to CSV
    df.to_csv(f'{out_path}/{task_id}/events.csv', index=False)
    # Release output video writer
    output_video.release()

if __name__ == '__main__':
    video_classifier(f'{path}/videos/event/redcard.mp4', 'redcard')