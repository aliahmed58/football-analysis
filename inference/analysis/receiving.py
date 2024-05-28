import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from inference.util import utils
from inference.firebase import firestore

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calc_receiving(file_path: str) -> pd.DataFrame:
    print('Calculating receiving....')
    df = pd.read_csv(file_path)

    team_counts = df['teamId'].value_counts()
    home_team = team_counts.idxmax()  # Most occurring team
    away_team = team_counts.index[1]  # Second most occurring team
    df['teamId'] = df['teamId'].apply(lambda x: 'Home' if x == home_team else ('Away' if x == away_team else 'Other'))

    # Radius for pressure calculation
    radius = 5
    threshold=10
    # Initialize lists to store x and y points
    xpoints = []
    ypoints = []
    pressures = []
    p=[]
    ts=[]
    team=[]
    c=0
    points = []
    epoints = []  
    start_x = None
    start_y = None
    end_y = None
    end_x = None  
    pass_completions = []
    # Iterate ove
    # r each row in the DataFrame
    for index, row in df.iterrows():
        if row['ball_posession'] == 1:
            # Reset the pressures list for each ball possession event
            pressures = []
            if start_x is None:  # Starting point
                start_x = row['x']
                start_y = row['y']
                points.append((start_x, start_y))  # Append starting point
                team.append(row['teamId'])
                ts.append(row['timestamp'])
                s=row['teamId']
                if end_x and end_y is not None:
                    epoints.append((end_x, end_y))      # Append ending point
                    e=row['teamId']
                else:
                    epoints.append((start_x, start_y))  # Append starting point
                    pass_completions.append("no pass attempted")
            else:  # Ending point
                end_x = row['x']
                end_y = row['y']
                start_x = None
                start_y = None
                # Get the x, y coordinates of the ball possession event
                ball_pos_x = row['x']
                ball_pos_y = row['y']
                frame = row['frame']
                distances = []
                for _, event_row in df[(df['frame'] == frame) & (df['ball_posession'] == 0)].iterrows():
                    event_x = event_row['x']
                    event_y = event_row['y']
                    distance = euclidean_distance((ball_pos_x, ball_pos_y), (event_x, event_y))
                    
                    # Calculate pressure rating considering the radius
                    if distance <= radius:
                        # Check if the event belongs to the opposite team
                        if event_row['teamId'] != row['teamId']:
                            if distance == 5: 
                                pressures.append(0.3)
                            elif distance<5:
                                pressures.append(max(0, radius - distance))  # Subtract distance from radius and append to pressures list
                            
                
                # Calculate the average pressure rating for the frame
                if pressures:
                    avg_pressure = max(pressures) # Average pressure if multiple players detected
                    p.append(round(avg_pressure))  # Append average pressure to pressures list
                else: 
                    p.append(0)
    
    if len(p) > len(points):
        p.pop()
    elif len(p) < len(points):
        p.append(0)
    # Create DataFrame df_pressure
    df_pressure = pd.DataFrame({
        "start": points,
        "end": epoints,
        "PressureRating": p,  # Use the pressures list
        "Team": team,
        "TimeStamp": ts,
    })
    
    # Threshold distance
    threshold = 10

    # Iterate through each row and apply distance threshold if PassCompletion is not "no pass attempted"
    for index, row in df_pressure.iterrows():
                distance = euclidean_distance(row['start'], row['end'])
                if distance < threshold:
                    # Remove row if distance is below threshold
                    df_pressure.drop(index, inplace=True)

    #find maxium ycoordinate in both starting and ending point in df_fin
    max_y = max(df_pressure['start'].apply(lambda x: x[1]).max(), df_pressure['end'].apply(lambda x: x[1]).max())
    max_y
    over= max_y-80
    df_pressure['start']=df_pressure['start'].apply(lambda x: (x[0],x[1]-over))
    df_pressure['end']=df_pressure['end'].apply(lambda x: (x[0],x[1]-over))

    # Filter out passes where the starting point of the current pass is the same as the ending point of the previous pass
    filtered_rows = []
    previous_end_point = None

    for index, row in df_pressure.iterrows():
            if previous_end_point is None or row['start'] != previous_end_point:
                filtered_rows.append(row)
            previous_end_point = row['end']

    filtered_df = pd.DataFrame(filtered_rows)

    filtered_df['start_x'] = filtered_df['start'].apply(lambda x: x[0])
    filtered_df['start_y'] = filtered_df['start'].apply(lambda x: x[1])
    filtered_df['end_x'] = filtered_df['end'].apply(lambda x: x[0])
    filtered_df['end_y'] = filtered_df['end'].apply(lambda x: x[1])

    # filtered_df.drop(columns=['start', 'end'], inplace=True)
    return filtered_df

def create_receiving_map(df_pressure: pd.DataFrame, side: str, task_id: str, data: dict):
    # Filter out passes where the starting point of the current pass is the same as the ending point of the previous pass
    filtered_rows = []
    previous_end_point = None

    for index, row in df_pressure.iterrows():
        if previous_end_point is None or row['start'] != previous_end_point:
            filtered_rows.append(row)
        previous_end_point = row['end']

    filtered_df = pd.DataFrame(filtered_rows)

    # Assuming you have already created a plot named 'ax'
    pitch = Pitch(pitch_color='green', goal_type='box', goal_alpha=1)
    fig, ax = pitch.draw()

    # Assigning colors based on PassCompletion
    pass_completion_colors = {
    "no pass attempted": "blue",
    "pass completed": "yellow",
    "pass incomplete": "red"
    }

    c=0
    # Iterate through the DataFrame to plot arrows between consecutive points
    for index, row in filtered_df.iterrows():
        if row['Team'] == side: #side define here
            x_start, y_start = row['start']
            x_end, y_end = row['end']
            marker_size = 10+(row['PressureRating']*30)
            ax.annotate("", xy=(x_start, y_start), xytext=(x_end, y_end),
                        arrowprops=dict(arrowstyle='<-', color='blue', alpha=0.5))
            # ax.scatter(x_start, y_start, color='blue', s=20, zorder=3, alpha=0.7)
            ax.scatter(x_end, y_end, color='RED', s=marker_size, zorder=3, alpha=0.7)
            c=c+1

    #poistion of highest pressure
    for index, row in df_pressure.iterrows():
        if row['PressureRating'] == max(df_pressure['PressureRating']):
            x, y = row['end']

    #reposndse for dashboard kpi
    print(c)
    print(x,y)       
    ax.text(40, 5, "Pressure when Passes Received", fontsize=12, color='black', weight='bold')


    out_dir = f'{utils.get_project_root()}/out/{task_id}/receiving_{side}.png'
    plt.savefig(out_dir)

    url = firestore.upload_file_to_firebase(
        out_dir,
        f'{side}/receiving.png',
        task_id
    )

    data[side]['images']['receiving'] = url
    
    data[side]['x_max_pressure'] = float(x)
    data[side]['y_max_pressure'] = float(y)

