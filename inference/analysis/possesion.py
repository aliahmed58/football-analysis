import pandas as pd

def calc_possession(file_name: str) -> pd.DataFrame:
    print('Calculating possession....')
    df = pd.read_csv(file_name)

    # Initialize lists to store x and y points
    xpoints = []
    ypoints = []
    ts=[]
    team=[]
    c=0
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        if row['ball_posession'] == 1:
            # Get the x, y coordinates of the ball possession event
            ball_pos_x = row['x']
            ball_pos_y = row['y']
            
            # Append x and y points
            xpoints.append(ball_pos_x)
            ypoints.append(ball_pos_y)
            team.append(row['teamId'])
            ts.append(row['timestamp'])
            
    # Create DataFrame df_pressure
    df_heatmap = pd.DataFrame({
        "x_Point": xpoints,
        "y_Point": ypoints,
        "Team": team,
        "TimeStamp": ts
    })

    df_heatmap['Team'].value_counts()

    #find maxium ycoordinate in both starting and ending point in df_fin
    max_y = df_heatmap['y_Point'].max()
    over= max_y-80

    df_heatmap['y_Point'] = df_heatmap['y_Point']-over

    return df_heatmap

