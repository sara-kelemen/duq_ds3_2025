import pandas as pd
import sqlite3
import random
import numpy as np
import fasttext # I need help getting fast text to work
import jarowinkler as jw
from itertools import product
from reusable_classifier import ReusableClassifier  

# Load FastText model once globally
ft_model = fasttext.load_model('?')


def read_doctors(db_path: str) -> pd.DataFrame:
    """
    Reads unique doctor (assignor) names from the database.
    """
    query = """
    SELECT DISTINCT surname, forename 
    FROM assignors 
    WHERE forename IS NOT NULL AND surname IS NOT NULL;
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    df["full_name"] = df["forename"] + " " + df["surname"]
    return df[["full_name"]]


def read_patentees(db_path: str) -> pd.DataFrame:
    """
    Reads unique assignee (patentee) names from the database.
    """
    query = """
    SELECT DISTINCT surname, forename 
    FROM assignees 
    WHERE forename IS NOT NULL AND surname IS NOT NULL;
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    df["full_name"] = df["forename"] + " " + df["surname"]
    return df[["full_name"]]


def read_manual_data(csv_path: str) -> pd.DataFrame:
    """
    Reads labeled training data for doctor-patentee name pairs.
    """
    return pd.read_csv(csv_path)


def simulate(real_name: str, delta: float) -> str:
    """
    Simulates a name variant by introducing character replacements.
    """
    real_name = real_name.lower()
    if not real_name or delta <= 0:
        return real_name
    name = list(real_name)
    change = max(1, int(delta * len(name)))
    for _ in range(change):
        i = random.randrange(len(name))
        name[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(name)


def create_training(dr_df: pd.DataFrame, pat_df: pd.DataFrame, manual_train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates all possible doctor-patentee name pairs and merges with manual labels.
    """
    pairs = list(product(dr_df["full_name"], pat_df["full_name"]))
    paired_name_df = pd.DataFrame(pairs, columns=["Doctor", "Patentee"])
    merged = paired_name_df.merge(manual_train_df, how='left',
                                  left_on=["Doctor", "Patentee"],
                                  right_on=["npi_name", "patent_name"])
    merged["label"] = merged["label"].fillna(0)
    return merged


def calc_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Jaro-Winkler and FastText-based distances for each doctor-patentee pair.
    """
    df[['npi_forename', 'npi_surname']] = df['Doctor'].str.lower().str.split(' ', 1, expand=True)
    df[['patent_forename', 'patent_surname']] = df['Patentee'].str.lower().str.split(' ', 1, expand=True)
    df['npi_vec'] = df['Doctor'].apply(lambda x: ft_model.get_sentence_vector(x))
    df['patent_vec'] = df['Patentee'].apply(lambda x: ft_model.get_sentence_vector(x))
    df['jw_dist_surname'] = df.apply(lambda r: jw.jaro_similarity(r['npi_surname'], r['patent_surname']), axis=1)
    df['jw_dist_forename'] = df.apply(lambda r: jw.jaro_similarity(r['npi_forename'], r['patent_forename']), axis=1)
    df['ft_dist_last_name'] = df.apply(lambda r: np.linalg.norm(r['npi_vec'] - r['patent_vec']), axis=1)
    return df


def train_model(df: pd.DataFrame, labels: pd.Series, model_type: str = 'random_forest') -> ReusableClassifier:
    """
    Trains a classifier using the distance-based features and provided labels.
    """
    clf = ReusableClassifier(model_type)
    features = df[['jw_dist_surname', 'jw_dist_forename', 'ft_dist_last_name']]
    clf.train(features, labels)
    return clf


def predict(model: ReusableClassifier, df: pd.DataFrame) -> pd.Series:
    """
    Uses a trained classifier to predict match labels for doctor-patentee pairs.
    """
    features = df[['jw_dist_surname', 'jw_dist_forename', 'ft_dist_last_name']]
    return model.predict(features)


def full_train_pipeline(dr_df: pd.DataFrame, pat_df: pd.DataFrame, manual_df: pd.DataFrame, model_type: str = 'xgboost'):
    """
    Runs the full name-matching pipeline: pairing, labeling, distance calc, training.
    """
    train_df = create_training(dr_df, pat_df, manual_df)
    feat_df = calc_distances(train_df)
    model = train_model(feat_df, train_df['label'], model_type)
    return model, feat_df
