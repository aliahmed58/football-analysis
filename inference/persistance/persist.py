import sqlalchemy as sa
from google.cloud.sql.connector import Connector
# db keys should be created with your own configurations
from inference.persistance.dbkeys import *
import pandas as pd
import traceback

def get_cloud_conn():
    connector = Connector()
    conn = connector.connect(
        CONNECTION_NAME,
        'pymysql',
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME
    )
    return conn

def get_engine() -> sa.Engine:
    engine = sa.create_engine(
        "mysql+pymysql://",
        creator=get_cloud_conn,
        echo=True
    )
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
    df = pd.read_csv('../out/4de6fcdc-5850-41e8-8e22-bf2dec5c8559/players.csv')
    df.to_sql('Test', con=get_engine(), if_exists='append')
    