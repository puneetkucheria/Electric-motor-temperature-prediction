import pandas as pd
import sqlite3


def get_data():
    con = sqlite3.connect('./../../Database.db')
    df = pd.read_sql_query("SELECT * FROM Electric_cars", con)
    return df

def fields_to_numeric(df):
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    return df

def clean_data(df):
    df.dropna(subset=['profile_id'],inplace=True)

    for id in df['profile_id'].unique():
        mask = (df['profile_id']==id)
        df.loc[mask,'u_d'] = df.loc[mask,'u_d'].mean()

    for id in df['profile_id'].unique():
        mask = (df['profile_id']==id)
        df.loc[mask,'motor_speed'] = df.loc[mask,'motor_speed'].mean()

    return df