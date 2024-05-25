import pandas as pd
import numpy as np

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calc_receiving(file_path: str) -> pd.DataFrame:
    print('Calculating receiving....')
    df = pd.read_csv(file_path)
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

    filtered_df.drop(columns=['start', 'end'], inplace=True)
    return filtered_df
