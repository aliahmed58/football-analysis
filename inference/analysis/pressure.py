import pandas as pd
import numpy as np

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calc_pressure(file_path: str) -> pd.DataFrame:
    print('Calculating pressure....')
    df = pd.read_csv(file_path)
    # Radius for pressure calculation
    radius = 5
    # Initialize lists to store x and y points
    xpoints = []
    ypoints = []
    pressures = []
    p=[]
    ts=[]
    team=[]
    c=0
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        if row['ball_posession'] == 1:
            # Reset the pressures list for each ball possession event
            pressures = []
            
            # Get the x, y coordinates of the ball possession event
            ball_pos_x = row['x']
            ball_pos_y = row['y']
            frame = row['frame']
            
            # Append x and y points
            xpoints.append(ball_pos_x)
            ypoints.append(ball_pos_y)
            team.append(row['teamId'])
            ts.append(row['timestamp'])
            # Calculate distances from the ball possession event to all other events in the same frame
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

    # Create DataFrame df_pressure
    df_pressure = pd.DataFrame({
        "x_Point": xpoints,
        "y_Point": ypoints,
        "PressureRating": p,  # Use the pressures list
        "Team": team,
        "TimeStamp": ts
    })

    #find maxium ycoordinate in both starting and ending point in df_fin
    max_y = df_pressure['y_Point'].max()
    over= max_y-80

    df_pressure['y_Point'] = df_pressure['y_Point']-over
    

    return df_pressure

