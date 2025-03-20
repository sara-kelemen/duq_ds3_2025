import pandas as pd
import sqlite3 
import random
from itertools import product

def read_doctors(db_path: str) -> pd.DataFrame:
    """
    Reads unique doctor (assignor) names from the database.
    
    Args:
        db_path (str): Path to the SQLite database.
    
    Returns:
        pd.DataFrame: A DataFrame with doctor names.
    """

    query = """SELECT DISTINCT surname, forename 
               FROM assignors 
               WHERE forename IS NOT NULL AND surname IS NOT NULL;"""
    
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Combine forename and surname into full name
        df["full_name"] = df["forename"] + " " + df["surname"]
        
        return df[["full_name"]]
    
    except Exception as e:
        print(f"Error reading from database: {e}")
        return pd.DataFrame()


def read_patentees(db_path: str) -> pd.DataFrame:
     """
    Reads unique assignee names from the database.
    
    Args:
        db_path (str): Path to the SQLite database.
    
    Returns:
        pd.DataFrame: A DataFrame with patentee names.
    """
     query = """SELECT DISTINCT surname, forename 
               FROM assignees 
               WHERE forename IS NOT NULL AND surname IS NOT NULL;"""
     try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Combine forename and surname into full name
        df["full_name"] = df["forename"] + " " + df["surname"]
        
        return df[["full_name"]]
     
     except Exception as e:
        print(f"Error reading from database: {e}")
        return pd.DataFrame()

def simulate(real_name: str, delta: float) -> str:
    """
    Simulates a variant of the real name by introducing typos and modifications.

    Args:
        real_name (str): The original name to modify.
        delta (float): Degree of modification (0.0 = no change, 1.0 = heavy change).

    Returns:
        str: The modified name.
    """
    real_name = real_name.lower()
    if not real_name or delta <= 0:
        return real_name  # No change if delta is 0 or name is empty
    
    name = list(real_name)
    length = len(name)

    # accounting for delta
    change = max(1, int(delta * length))  # At least one change

    for i in range(change):
        choice = random.randrange(length)
        new_char = random.choice("abcdefghijklmnopqrstuvwxyz")
        name[choice] = new_char # conduct replacement
    
    return "".join(name)

def read_manual_data(csv_path: str) -> pd.DataFrame:
    manual_train_df = pd.read_csv(csv_path)
    return manual_train_df

def create_training(dr_df, pat_df, manual_train_df) -> pd.DataFrame:
    pairs = list(product(dr_df, pat_df))
    paired_name_df = pd.DataFrame(pairs, columns=["Doctor", "Patentee"])
    return paired_name_df

def calc_distances(paired_name_df: pd.DataFrame) -> features_df: pd.DataFrame:

    
def train_model(features_df, labels_series: pd.Series) -> ReusableClassifier()

def pairwise(dr_df, pat_df) -> pair_name_df: pd.DataFrame:

calc_distances -> predict
  # identify a set of training rows to save as csv 