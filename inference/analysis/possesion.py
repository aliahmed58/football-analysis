import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Arc
from inference.util import utils
from inference.firebase import firestore

def calc_possession(file_name: str) -> pd.DataFrame:
    print('Calculating possession....')
    df = pd.read_csv(file_name)

    team_counts = df['teamId'].value_counts()
    home_team = team_counts.idxmax()  # Most occurring team
    away_team = team_counts.index[1]  # Second most occurring team
    df['teamId'] = df['teamId'].apply(lambda x: 'Home' if x == home_team else ('Away' if x == away_team else 'Other'))

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

def create_heatmap(df_heatmap: pd.DataFrame, side: str, task_id: str, data: dict) -> None:
    #Create figure
    fig=plt.figure()
    fig.set_size_inches(7, 5)
    ax=fig.add_subplot(1,1,1)

    #Pitch Outline & Centre Line
    plt.plot([0,0],[0,90], color="black")
    plt.plot([0,130],[90,90], color="black")
    plt.plot([130,130],[90,0], color="black")
    plt.plot([130,0],[0,0], color="black")
    plt.plot([65,65],[0,90], color="black")

    #Left Penalty Area
    plt.plot([16.5,16.5],[65,25],color="black")
    plt.plot([0,16.5],[65,65],color="black")
    plt.plot([16.5,0],[25,25],color="black")

    #Right Penalty Area
    plt.plot([130,113.5],[65,65],color="black")
    plt.plot([113.5,113.5],[65,25],color="black")
    plt.plot([113.5,130],[25,25],color="black")

    #Left 6-yard Box
    plt.plot([0,5.5],[54,54],color="black")
    plt.plot([5.5,5.5],[54,36],color="black")
    plt.plot([5.5,0.5],[36,36],color="black")

    #Right 6-yard Box
    plt.plot([130,124.5],[54,54],color="black")
    plt.plot([124.5,124.5],[54,36],color="black")
    plt.plot([124.5,130],[36,36],color="black")

    #Prepare Circles
    centreCircle = plt.Circle((65,45),9.15,color="black",fill=False)
    centreSpot = plt.Circle((65,45),0.8,color="black")
    leftPenSpot = plt.Circle((11,45),0.8,color="black")
    rightPenSpot = plt.Circle((119,45),0.8,color="black")

    #Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)

    #Prepare Arcs
    leftArc = Arc((11,45),height=18.3,width=18.3,angle=0,theta1=310,theta2=50,color="black")
    rightArc = Arc((119,45),height=18.3,width=18.3,angle=0,theta1=130,theta2=230,color="black")

    #Draw Arcs
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)

    #Tidy Axes
    plt.axis('off')
    df_home = df_heatmap[df_heatmap['Team'] == side] #side tell here
    sns.kdeplot(x=df_home['x_Point'], y=df_home['y_Point'], shade=True,n_levels=50)
    # sns.kdeplot(x=df_heatmap['x_Point'], y=df_heatmap['y_Point'], shade=True,n_levels=50)
    plt.xlim(0, 120)
    plt.ylim(80, 0)  # Reverse the y-axis

    ax.text(40, 5, "Possesion HeatMap", fontsize=12, color='black', weight='bold')

    #avg points of home/away team: dshbaord content
    avg_x = df_home['x_Point'].mean()
    avg_y = df_home['y_Point'].mean()
    print(avg_x, avg_y)
    # Save the plot as an image
    fig.savefig('soccer_pitch_plot.png', dpi=300, bbox_inches='tight')

    out_dir = f'{utils.get_project_root()}/out/{task_id}/possession_{side}.png'
    plt.savefig(out_dir)

    url = firestore.upload_file_to_firebase(
        out_dir,
        f'{side}/possession.png',
        task_id
    )

    data[side]['images']['possession'] = url

    data[side]['avg_x'] = float(avg_x)
    data[side]['avg_y'] = float(avg_y)
    

