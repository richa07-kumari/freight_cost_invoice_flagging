import sqlite3
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_vendor_invoice_data(db_path: str):
    """
    Load vendor invoice data from sqlite database
    """
    conn=sqlite3.connect(db_path)
    query="SELECT * FROM vendor_invoice"
    df=pd.read_sql_query(query,conn)
    conn.close()
    return df

def feature_engineering(df: pd.DataFrame):
    """
    Create new features
    """
    # Log transform Dollars to reduce skewness
    df['Dollar_transformed'] = np.log1p(df['Dollars'].clip(lower=0))

    return df

def prepare_features(df: pd.DataFrame):
    """
    select features and target variable
    """
    X=df[["Dollar_transformed"]]
    y=df["Freight"]
    return X,y

def split_data(X,y,test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets
    """
    return train_test_split(
        X,y, test_size=test_size, random_state=random_state
    )
    