import jarowinkler as jw
import pandas as pd
import random
from fuzzywuzzy import fuzz

from duq_ds3_2025.entity_res import db
from duq_ds3_2025.entity_res import name_classifier


class NameMatcher:
    def __init__(self, db_path: str):
        """
        Initialize the pipeline with database path, FastText model, and classifier type.
        """
        self.db_path = db_path
        self.model = None

    def read_doctors(self) -> pd.DataFrame:
        """Read doctor data from sqlite database"""
        query = """
        SELECT DISTINCT id, LOWER(surname) AS surname, LOWER(forename) AS forename
        FROM npi_providers
        WHERE forename IS NOT NULL;
        """
        conn = db.db(self.db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def read_patentees(self) -> pd.DataFrame:
        """Read patentee data from sqlite database"""
        query = """
        SELECT DISTINCT patent_id AS id, LOWER(name) as name
        FROM assignees 
        WHERE name IS NOT NULL;
        """
        conn = db.db(self.db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        df['surname'] = df['name'].str.split(' ').apply(
            lambda x: x[-1])
        df['forename'] = df['name'].str.split(' ').apply(
            lambda x: ' '.join(x[:-1]) if len(x) > 1 else None)
        
        df = df.dropna().reset_index(drop=True)
        return df
    
    def simulate_typos_in_name(self, real_name: str, delta: float) -> str:
        """Simulate typos to create training data from a single name

        Args:
            real_name (str): real name
            delta (float): number of errors to add

        Returns:
            str: error-filled name
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
    
    def create_manual_training_data(self, 
                                    num_examples: int = 30) -> pd.DataFrame:
        """
        Create more balanced simulated training data using multiple positives 
        and negatives per doctor.
        """
        dr_df = self.read_doctors().drop(columns=['id']).sample(n=num_examples, replace=False)
        pat_df = self.read_patentees().drop(columns=['id']).sample(n=num_examples, replace=False)
        names = pd.concat([dr_df, pat_df])

        manual_examples = []
        for _, dr in dr_df.iterrows():
            for _, pat in pat_df.iterrows():
                manual_examples.append({
                    'dr_forename': dr['forename'],
                    'dr_surname': dr['surname'],
                    'pat_forename': pat['forename'],
                    'pat_surname': pat['surname'],
                    'label': 0
                })

        for _, name in names.iterrows():
            manual_examples.append({
                    'dr_forename': name['forename'],
                    'dr_surname': name['surname'],
                    'pat_forename': self.simulate_typos_in_name(name['forename'], delta=0.3),
                    'pat_surname': self.simulate_typos_in_name(name['surname'], delta=0.3),
                    'label': 1
                })
            
        return pd.DataFrame(manual_examples)

    def calc_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the distances between dr and patentee names.
        Return ONLY the distances and an optional label column.
        """
        assert 'dr_forename' in df
        assert 'pat_forename' in df

        df['jw_dist_surname'] = df.apply(lambda r: jw.jaro_similarity(str(r['dr_surname']), str(r['pat_surname'])), axis=1)
        df['jw_dist_forename'] = df.apply(lambda r: jw.jaro_similarity(str(r['dr_forename']), str(r['pat_forename'])), axis=1)
        df['lev_dist_surname'] = df.apply(lambda r: fuzz.ratio(str(r['dr_surname']), str(r['pat_surname'])), axis=1)
        df['lev_dist_forename'] = df.apply(lambda r: fuzz.ratio(str(r['dr_forename']), str(r['pat_forename'])), axis=1)

        cols = ['jw_dist_surname',
                'jw_dist_forename',
                'lev_dist_surname',
                'lev_dist_forename']
        if 'label' in df:
            cols.append('label')

        return df[cols]
    
    def train(self):
        """
        Full training pipeline with simulated labels.
        """
        train_df = self.create_manual_training_data()

        # Balance the dataset before feature calc
        pos = train_df[train_df["label"] == 1]
        neg = train_df[train_df["label"] == 0].sample(n=len(pos), random_state=42)
        train_df = pd.concat([pos, neg]).reset_index(drop=True)

        feat_df = self.calc_distances(train_df)
        labels = feat_df['label']
        feat_df = feat_df.drop(columns=['label'])

        self.model = name_classifier.NameClassifier('random_forest')
        self.model.train(feat_df, labels)
        print("Model training complete!")

    def blocks(self):
        """Create blocks of test data, yielding combined features + IDs"""
        dr_df = self.read_doctors()
        pat_df = self.read_patentees()

        # Simplest blocking possible: first letter of last name
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            dr_letter = dr_df.loc[dr_df['surname'].str[0] == letter]
            pat_letter = pat_df.loc[pat_df['surname'].str[0] == letter]

            combos = []
            ids = []
            for _, dr in dr_letter.iterrows():
                for _, pat in pat_letter.iterrows():
                    combos.append({
                        'dr_forename': dr['forename'],
                        'dr_surname': dr['surname'],
                        'pat_forename': pat['forename'],
                        'pat_surname': pat['surname'],
                    })
                    ids.append({'dr_id': dr['id'], 'pat_id': pat['id']})

            if combos:  # only proceed if there's data
                feature_df = self.calc_distances(pd.DataFrame(combos)).reset_index(drop=True)
                ids_df = pd.DataFrame(ids).reset_index(drop=True)
                full_df = pd.concat([feature_df, ids_df], axis=1)
                yield full_df


    def predict_matches(self):
        """Run the backlog of doctor to patentee matches"""
        conn = db.db(self.db_path)
        for full_df in self.blocks():
            full_df['label'] = self.model.predict(full_df[['jw_dist_surname', 'jw_dist_forename', 'lev_dist_surname', 'lev_dist_forename']])

            matches = full_df[full_df['label'] == 1]
            print(f"{len(matches)} matches found")

            for _, row in matches.iterrows():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO doctor_patentee_matches (dr_id, pat_id)
                    VALUES (?, ?)
                    """,
                    (row['dr_id'], row['pat_id'])
                )
        conn.close()




if __name__ == '__main__':
    pipeline = NameMatcher(db_path='data/patent_npi_db.sqlite')
    pipeline.train()

    # Save the trained model
    pipeline.model.save('data/name_matcher_model_2')

    pipeline.predict_matches()

