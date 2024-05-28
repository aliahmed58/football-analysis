import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
from inference.util import utils
from inference.firebase import firestore

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calc_pressure(file_path: str) -> pd.DataFrame:
    print('Calculating pressure....')
    df = pd.read_csv(file_path)

    team_counts = df['teamId'].value_counts()
    home_team = team_counts.idxmax()  # Most occurring team
    away_team = team_counts.index[1]  # Second most occurring team
    df['teamId'] = df['teamId'].apply(lambda x: 'Home' if x == home_team else ('Away' if x == away_team else 'Other'))

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


def create_pressure_map(df_pressure: pd.DataFrame, side: str, task_id: str, data: dict):
    print('Saving pressure map....')
    pitch = Pitch(line_zorder=2)
    fig, ax = pitch.draw()
    df_home = df_pressure[df_pressure['Team'] == side] #side decide here
    # Calculate the bin statistic with weights based on pressure rating
    bin_statistic = pitch.bin_statistic(
        df_home['x_Point'], df_home['y_Point'], df_home['PressureRating'], bins=(10, 5)
    )

    # Gaussian smoothing of the bin statistic
    bin_statistic['statistic'] = gaussian_filter(bin_statistic['statistic'], sigma=1)

    # Plot the smoothed heatmap
    heatmap = pitch.heatmap(bin_statistic, ax=ax, cmap='Reds', edgecolor='#f9f9f9', alpha=0.5)
    ax.text(45, 0, "Pressure Rating", fontsize=12, color='black', weight='bold')

    #resposne for Dashbaord KPi
    avg_pressure = df_home['PressureRating'].mean()
    min_pressure = df_home['PressureRating'].min()
    max_pressure = df_home['PressureRating'].max()

    out_dir: str = f'{utils.get_project_root()}/out/{task_id}/pressure_{side}.png'
    plt.savefig(out_dir)

    url = firestore.upload_file_to_firebase(
        out_dir,
        f'{side}/pressure.png',
        task_id
    )

    data[side]['images']['pressure'] = url
    
    data[side]['avg_pressure'] = float(avg_pressure)
    data[side]['min_pressure'] = float(min_pressure)
    data[side]['max_pressure'] = float(max_pressure)



