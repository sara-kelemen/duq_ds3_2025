import pandas as pd
import sqlite3
import random
import numpy as np
import fasttext
import jarowinkler as jw
from itertools import product
from reusable_classifier import ReusableClassifier


class NameMatcherPipeline:
    def __init__(self, db_path: str, ft_model_path: str, model_type: str = 'xgboost'):
        """
        Initialize the pipeline with database path, FastText model, and classifier type.
        """
        self.db_path = db_path
        self.model_type = model_type
        self.ft_model = fasttext.load_model(ft_model_path)
        self.model = None
        self.dr_df = None
        self.pat_df = None

    def read_doctors(self) -> pd.DataFrame:
        query = """
        SELECT DISTINCT surname, forename 
        FROM assignors 
        WHERE surname IS NOT NULL AND forename IS NOT NULL;
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        df["full_name"] = df["forename"] + " " + df["surname"]
        return df[["full_name"]]

    def read_patentees(self) -> pd.DataFrame:
        query = """
        SELECT DISTINCT name
        FROM assignees 
        WHERE name IS NOT NULL;
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        df["full_name"] = df["name"]
        return df[["full_name"]]

    def simulate(self, real_name: str, delta: float) -> str:
        real_name = real_name.lower()
        if not real_name or delta <= 0:
            return real_name
        name = list(real_name)
        change = max(1, int(delta * len(name)))
        for _ in range(change):
            i = random.randrange(len(name))
            name[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
        return "".join(name)

    def create_manual_data(self, dr_df: pd.DataFrame, num_examples: int = 30) -> pd.DataFrame:
        manual_examples = []
        for i in range(min(num_examples, len(dr_df))):
            real_name = dr_df.iloc[i]["full_name"]
            manual_examples.append({"npi_name": real_name, "patent_name": real_name, "label": 1})
            sim_name = self.simulate(real_name, delta=0.3)
            manual_examples.append({"npi_name": real_name, "patent_name": sim_name, "label": 0})
        return pd.DataFrame(manual_examples)

    def create_training(self, dr_df: pd.DataFrame, pat_df: pd.DataFrame, manual_train_df: pd.DataFrame) -> pd.DataFrame:
        pairs = list(product(dr_df["full_name"], pat_df["full_name"]))
        paired_name_df = pd.DataFrame(pairs, columns=["Doctor", "Patentee"])
        merged = paired_name_df.merge(manual_train_df, how='left',
                                      left_on=["Doctor", "Patentee"],
                                      right_on=["npi_name", "patent_name"])
        merged["label"] = merged["label"].fillna(0)
        return merged

    def calc_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        df[['npi_forename', 'npi_surname']] = df['Doctor'].str.lower().str.split(' ', 1, expand=True)
        df[['patent_forename', 'patent_surname']] = df['Patentee'].str.lower().str.split(' ', 1, expand=True)
        df['npi_vec'] = df['Doctor'].apply(lambda x: self.ft_model.get_sentence_vector(x))
        df['patent_vec'] = df['Patentee'].apply(lambda x: self.ft_model.get_sentence_vector(x))
        df['jw_dist_surname'] = df.apply(lambda r: jw.jaro_similarity(r['npi_surname'], r['patent_surname']), axis=1)
        df['jw_dist_forename'] = df.apply(lambda r: jw.jaro_similarity(r['npi_forename'], r['patent_forename']), axis=1)
        df['ft_dist_last_name'] = df.apply(lambda r: np.linalg.norm(r['npi_vec'] - r['patent_vec']), axis=1)
        return df

    def train(self):
        """
        Full training pipeline.
        """
        self.dr_df = self.read_doctors()
        self.pat_df = self.read_patentees()
        manual_df = self.create_manual_data(self.dr_df)

        train_df = self.create_training(self.dr_df, self.pat_df, manual_df)
        feat_df = self.calc_distances(train_df)

        self.model = ReusableClassifier(self.model_type)
        features = feat_df[['jw_dist_surname', 'jw_dist_forename', 'ft_dist_last_name']]
        self.model.train(features, train_df['label'])

        print("Model training complete!")
        print(feat_df[["Doctor", "Patentee", "label", "jw_dist_surname", "jw_dist_forename", "ft_dist_last_name"]].head())

    def predict_matches(self, output_path: str = "data/predicted_matches.csv"):
        """
        Predict actual doctor-patentee matches and save them.
        """
        pairs_df = pd.DataFrame(list(product(self.dr_df["full_name"], self.pat_df["full_name"])), columns=["Doctor", "Patentee"])
        feat_df = self.calc_distances(pairs_df)
        feat_df["predicted_match"] = self.model.predict(feat_df[['jw_dist_surname', 'jw_dist_forename', 'ft_dist_last_name']])

        matched = feat_df[feat_df["predicted_match"] == 1]
        print("\nTop predicted matches:")
        print(matched[["Doctor", "Patentee", "jw_dist_surname", "jw_dist_forename", "ft_dist_last_name"]].head())

        matched.to_csv(output_path, index=False)
        print(f"\nSaved predicted matches to {output_path}")



#  RUN PIPELINE


if __name__ == "__main__":
    pipeline = NameMatcherPipeline(
        db_path="data/patent_npi_db.sqlite",
        ft_model_path="models/cc.en.300.bin",
        model_type="xgboost"
    )

    pipeline.train()
    pipeline.predict_matches()
