import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch
from inference.util import utils
from inference.firebase import firestore

# ---------------------------------------------
# Util functions needed while processing
# ---------------------------------------------

# Coordinates of the goal
gc_left = (0, 40)
gc_right = (120, 0)

# Define the starting and ending rectangles
sixyard_left = [(0, 18), (18, 18), (18, 62), (0, 62)]
pen_left = [(0, 30), (5, 30), (5, 50), (0, 50)]
#right
sixyard_right = [(115, 30), (120, 30), (115, 50), (120, 50)]
pen_right=[(102, 18), (120, 18), (120, 62), (102, 62)]

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Calculate progressive pass
def is_progressive_pass(start, end, goal, threshold=0.25):
    start_distance = euclidean_distance(start, goal)
    end_distance = euclidean_distance(end, goal)
    return end_distance < start_distance * (1 - threshold)

# Check if a given point is in a rect
def point_in_rectangle(point, rect):
    x, y = point
    x_coords, y_coords = zip(*rect)
    return min(x_coords) <= x <= max(x_coords) and min(y_coords) <= y <= max(y_coords)

def determine_pass_type(row):

    if row['PassCompletion'] != 'no pass attempted' and row['Team'] != 'Other': #chekc how many passes attempted were progressive, completed or incomplete, effective
        if row['Team'] == 'Home':
            goal_coordinates = gc_left
            starting_rect=pen_left
            ending_rect=sixyard_left
        elif row['Team'] == 'Away':
            goal_coordinates = gc_right
            starting_rect=pen_left
            ending_rect=sixyard_right
        if point_in_rectangle(row['Starting_Point'], starting_rect) and point_in_rectangle(row['Ending_Point'], ending_rect):
            return 'cutback'
        if is_progressive_pass(row['Starting_Point'], row['Ending_Point'], goal_coordinates):
            return 'progressive pass'
        if not point_in_rectangle(row['Starting_Point'], ending_rect) and point_in_rectangle(row['Ending_Point'], ending_rect):
            return 'cross'
        if euclidean_distance(row['Starting_Point'], row['Ending_Point']) > 32:  # 35 yards = ~32 meters
            return 'long ball'
    return 'Normal Pass'

def calc_passing(file_path: str) -> pd.DataFrame:
    print('Calculating passes....')
    df = pd.read_csv(file_path)

    team_counts = df['teamId'].value_counts()
    home_team = team_counts.idxmax()  # Most occurring team
    away_team = team_counts.index[1]  # Second most occurring team
    df['teamId'] = df['teamId'].apply(lambda x: 'Home' if x == home_team else ('Away' if x == away_team else 'Other'))

    points = []
    start_x = None
    start_y = None
    end_y = None
    end_x = None

    for index, row in df.iterrows():
        if row['ball_posession'] == 1:  # Check if ball possession is 1
            if start_x is None:  # Starting point
                start_x = row['x']
                start_y = row['y']
                points.append((start_x, start_y, row['teamId'], row['timestamp']))  # Append starting point
                if end_x and end_y is not None:
                    points.append((end_x, end_y, row['teamId'], row['timestamp']))      # Append ending point
            else:  # Ending point
                end_x = row['x']
                end_y = row['y']
                start_x = None
                start_y = None

    # Initialize lists to store data for DataFrame
    starting_points = []
    ending_points = []
    pass_completions = []
    team=[]
    ts=[]

    # Iterate through the points list
    for i in range(len(points) - 1):
        current_point = points[i]
        next_point = points[i + 1]

        # Extract coordinates and teamID
        current_coords = current_point[:2]
        next_coords = next_point[:2]
        team_id_current = current_point[2]
        team_id_next = next_point[2]

        # Determine pass completion
        if team_id_current == team_id_next:
            if current_coords == next_coords:
                pass_completions.append("no pass attempted")
                team.append(team_id_current)
            else:
                pass_completions.append("pass completed")
                team.append(team_id_current)
        else:
            pass_completions.append("pass incomplete")
            team.append(team_id_current)

        # Add starting and ending points to lists
        starting_points.append(current_coords)
        ending_points.append(next_coords)
        ts.append(current_point[3])

    # Create DataFrame
    df_fin = pd.DataFrame({
        "Starting_Point": starting_points,
        "Ending_Point": ending_points,
        "PassCompletion": pass_completions,
        "Team": team,
        "TS":ts
    })

    # Threshold distance
    threshold = 8

    # Iterate through each row and apply distance threshold if PassCompletion is not "no pass attempted"
    for index, row in df_fin.iterrows():
        if row['Team'] != 'Other':
            if row['PassCompletion'] != "no pass attempted":
                distance = euclidean_distance(row['Starting_Point'], row['Ending_Point'])
                if distance < threshold:
                    # Remove row if distance is below threshold
                    df_fin.drop(index, inplace=True)

    #find maxium ycoordinate in both starting and ending point in df_fin
    max_y = max(df_fin['Starting_Point'].apply(lambda x: x[1]).max(), df_fin['Ending_Point'].apply(lambda x: x[1]).max())
    over= max_y-80
    df_fin['Starting_Point']=df_fin['Starting_Point'].apply(lambda x: (x[0],x[1]-over))
    df_fin['Ending_Point']=df_fin['Ending_Point'].apply(lambda x: (x[0],x[1]-over))


    df_fin['TypeOfPass'] = df_fin.apply(determine_pass_type, axis=1)
    #break x and y coordinates into two columns
    df_fin['Starting_Point_X'] = df_fin['Starting_Point'].apply(lambda x: x[0])
    df_fin['Starting_Point_Y'] = df_fin['Starting_Point'].apply(lambda x: x[1])
    df_fin['Ending_Point_X'] = df_fin['Ending_Point'].apply(lambda x: x[0])
    df_fin['Ending_Point_Y'] = df_fin['Ending_Point'].apply(lambda x: x[1])

    # df_fin.drop(columns=['Starting_Point', 'Ending_Point'], inplace=True)

    return df_fin

def create_pass_map_complete(df_fin: pd.DataFrame, side: str, task_id: str, data: dict):
    # Filter out passes where the starting point of the current pass is the same as the ending point of the previous pass
    filtered_rows = []
    previous_end_point = None

    for index, row in df_fin.iterrows():
        if row['PassCompletion'] == "pass completed":
            print(row['Starting_Point'])
            if previous_end_point is None or row['Starting_Point'] != previous_end_point:
                filtered_rows.append(row)
            previous_end_point = row['Ending_Point']
            print(row['Ending_Point'])

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
    NormalPass=0
    longball=0
    progressive=0
    cutback=0
    cross=0
    # Iterate through the DataFrame to plot arrows between consecutive points
    for index, row in filtered_df.iterrows():
        if row['PassCompletion'] == "pass completed":
            if row['Team'] == side: #SIDE TELL HERE
                x_start, y_start = row['Starting_Point']     # Starting point
                x_end, y_end = row['Ending_Point']     # Ending point
                pass_completion = row['PassCompletion']  # Pass completion status
                # Plotting arrows and scatter points
                if row['TypeOfPass']=='Normal Pass':
                    NormalPass=NormalPass+1
                elif row['TypeOfPass']=='long ball':
                    longball=longball+1
                elif row['TypeOfPass']== 'progressive pass':
                    progressive = progressive+1
                elif row['TypeOfPass']=='cutback':  
                    cutback=cutback+1
                else:
                    cross=cross+1
                ax.annotate("", xy=(x_start, y_start), xytext=(x_end, y_end),
                        arrowprops=dict(arrowstyle='<-', color='blue'), alpha=0.5,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=1))
                ax.text(40, 5, "Passes Completed", fontsize=12, color='black', weight='bold')
                c=c+1

    #these things will be added on dashabrod send as reponse
    # ax.text(0, 0, f"Count of Team Red: {c}", fontsize=10, color='black')
    print("Normal Pass:", NormalPass)
    print("Long Ball:", longball)
    print("Progressive Pass:", progressive)
    print("Cutback:", cutback)
    print("Cross:", cross)

    out_dir = f'{utils.get_project_root()}/out/{task_id}/passes_complete_{side}.png'
    plt.savefig(out_dir)

    # save to firebase
    url = firestore.upload_file_to_firebase(
        out_dir, 
        f'{side}/passes_completed.png',
        f'detection/{task_id}'
    )

    data[side]['images']['passes_completed'] = url

    data[side]['passes']['complete'] = {
        'normal': int(NormalPass), 
        'long': int(longball),
        'progressive': int(progressive),
        'cutback': int(cutback),
        'cross': int(cross),
        'total_passes': int(c)
    }
    


def create_pass_map_incomplete(df_fin: pd.DataFrame, side: str, task_id: str, data: dict):

    # Assuming you have already created a plot named 'ax'
    pitch = Pitch(pitch_color='green', goal_type='box', goal_alpha=1)
    fig, ax = pitch.draw()

    # Assigning c
    # olors based on PassCompletion
    pass_completion_colors = {
        "no pass attempted": "blue",
        "pass completed": "yellow",
        "pass incomplete": "red"
    }
    c=0
    NormalPass=0
    longball=0
    progressive=0
    cutback=0
    cross=0
    # Iterate through the DataFrame to plot arrows between consecutive points
    for index, row in df_fin.iterrows():
        if row['Team'] == side: #side define here
            if row['PassCompletion'] == "pass incomplete":
                x_start, y_start = row['Starting_Point']     # Starting point
                x_end, y_end = row['Ending_Point']     # Ending point
                pass_completion = row['PassCompletion']  # Pass completion status
                if row['TypeOfPass']=='Normal Pass':
                    NormalPass=NormalPass+1
                elif row['TypeOfPass']=='long ball':
                    longball=longball+1
                elif row['TypeOfPass']== 'progressive pass':
                    progressive = progressive+1
                elif row['TypeOfPass']=='cutback':  
                    cutback=cutback+1
                else:
                    cross=cross+1
                # Plotting arrows and scatter points
                ax.arrow(x_start, y_start, x_end - x_start, y_end - y_start,
                        color=pass_completion_colors[pass_completion], alpha=1,
                        head_width=2, head_length=2, zorder=0)  # Adjust head width and length as needed
                ax.scatter(x_start, y_start, color=pass_completion_colors[pass_completion], zorder=1, alpha=0.1)
                ax.scatter(x_end, y_end, color=pass_completion_colors[pass_completion], zorder=1, alpha=0.1)
                # Add text annotation for "Passes Completed" heading and count of passes completed
                ax.text(40, 5, "Passes Incompleted", fontsize=12, color='black', weight='bold')
                c=c+1

    #these things will be added on dashabrod send as reponse
    # ax.text(0, 0, f"Count of Team Red: {c}", fontsize=10, color='black')
    #count or c also return
    print("Normal Pass:", NormalPass)
    print("Long Ball:", longball)
    print("Progressive Pass:", progressive)
    print("Cutback:", cutback)
    print("Cross:", cross)

    out_dir: str = f'{utils.get_project_root()}/out/{task_id}/passess_incomplete_{side}.png'
    plt.savefig(out_dir)

    url = firestore.upload_file_to_firebase(
        out_dir,
        f'{side}/passes_incomplete.png', 
        f'detection/{task_id}'
    )

    data[side]['images']['passes_incomplete'] = url

    data[side]['passes']['incomplete'] = {
        'normal': int(NormalPass), 
        'long': int(longball),
        'progressive': int(progressive),
        'cutback': int(cutback),
        'cross': int(cross),
        'total_passes': int(c)
    }