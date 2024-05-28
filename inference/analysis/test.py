from inference.analysis import passing, receiving, pressure, possesion
from inference.util import utils
from pprint import pprint

import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

task_id = 'manual-run'
input_file = f'{utils.get_project_root()}/out/{task_id}/players.csv'

df_passing = passing.calc_passing(input_file)
df_receiving = receiving.calc_receiving(input_file)
df_pressure = pressure.calc_pressure(input_file)
df_possession = possesion.calc_possession(input_file)

# Home team
sides = ['Home', 'Away']
# sample output json object, base dict that needs to be passed should be like this.
data: dict = {
    'Home': {
        'images': {

        },
        'passes': {

        }
    },
    'Away': {
        'images': {

        },
        'passes': {

        }
    }
}

for side in sides:
    # passing
    passing.create_pass_map_complete(df_passing, side, task_id, data)
    passing.create_pass_map_incomplete(df_passing, side, task_id, data)

    # receiving
    receiving.create_receiving_map(df_receiving, side, task_id, data)

    # pressure
    pressure.create_pressure_map(df_pressure, side, task_id, data)

    # possession
    possesion.create_heatmap(df_possession, side, task_id, data)

with open('test.json', 'w') as f:
    json.dump(data, f, cls=NpEncoder)