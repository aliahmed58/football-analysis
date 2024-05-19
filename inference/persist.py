import sqlalchemy as sa
from google.cloud.sql.connector import Connector
# db keys should be created with your own configurations
from dbkeys import *
import pandas as pd

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
        print(str(e))
        print('Failed whlie saving data to database')
    