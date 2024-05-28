import sqlalchemy as sa
import pandas as pd
import traceback
from inference.util import utils
from google.cloud.sql.connector import Connector
# db keys should be created with your own configurations
from inference.persistance.dbkeys import *

# def get_cloud_conn():
#     connector = Connector()
#     conn = connector.connect(
#         CONNECTION_NAME,
#         'pymysql',
#         user=DB_USER,
#         password=DB_PASS,
#         db=DB_NAME
#     )
#     return conn

def get_engine() -> sa.Engine:
    username = usernamemysql
    password = passwordmysql 
    server = servermysql
    database = databasemysql 

    engine = sa.create_engine(f'mysql+pymysql://{username}:{password}@{server}/{database}')

    return engine

def save_list_to_sql(data: list, engine: sa.Engine) -> bool:
    df = pd.DataFrame(data)
    try:
        df.to_sql(
        'Players',
        con=engine,
        if_exists='append'
        )
    except Exception as e:
        print(traceback.format_exc())
        print('Failed whlie saving data to database')

# Tests

def test_save_list_to_sql():
    data_list = [
        {"A": 1, "B": 2}
    ]    
    engine: sa.Engine = get_engine()
    save_list_to_sql(data_list, engine)

if __name__ == '__main__':
    engine = get_engine()

    from inference.analysis import passing, possesion, pressure, receiving
    file_path: str = f'{utils.get_project_root()}/out/manual-run/players.csv'
    df_passing: pd.DataFrame = passing.calc_passing(file_path)
    df_possesion: pd.DataFrame = possesion.calc_possession(file_path)
    df_receiving: pd.DataFrame = receiving.calc_receiving(file_path)
    df_pressure: pd.DataFrame = pressure.calc_pressure(file_path)

    # save these dataframes to sql
    df_passing.to_sql('d_passing', con=engine, if_exists='replace')
    df_possesion.to_sql('d_possession', con=engine, if_exists='replace')
    df_receiving.to_sql('d_receiving', con=engine, if_exists='replace')
    df_pressure.to_sql('d_pressure', con=engine, if_exists='replace')