import json
import os
import pandas as pd
import pickle
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import sklearn.preprocessing
# from xgboost import XGBClassifier  # Added XGBoost

from duq_ds3_2025 import reusable_classifier

class NameClassifier():
    def __init__(self, model_type: str):
        """Create a classifier, storing a model and metadata."""
        self.model_type = model_type
        if model_type == 'logistic_regression':
            self.model = self._create_logistic_regression()
        elif model_type == 'random_forest':
            self.model = self._create_random_forest()
        elif model_type == 'xgboost':  
            self.model = self._create_xgboost()
        else:
            raise ValueError('No model implemented of that type')

        self.scaler = None
        self.metadata = None

    def train(self, features: pd.DataFrame, labels: pd.Series, test_frac: float = 0.1):
        """Train the model from pandas data."""
        self._assess_tf_fraction(labels)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(features)
        features = self.scaler.transform(features)

        features_train, features_test, labels_train, labels_test = \
            sklearn.model_selection.train_test_split(features, labels, test_size=test_frac)

        self.model.fit(features_train, labels_train)
        pred_labels = self.model.predict(features_test)

        accuracy = (pred_labels == labels_test).mean()
        self.metadata = {'training_rows': len(features_train), 'accuracy': accuracy, 'model_type': self.model_type}
        print('Accuracy on test set is ', accuracy)

    def predict(self, features: pd.DataFrame):
        """Predict labels from features."""
        features = self.scaler.transform(features)
        return self.model.predict(features)

    def save(self, path: str):
        """Save model, scaler, and metadata."""
        model_path, _ = os.path.splitext(path)
        with open(model_path + '.pkl', 'wb') as fp:
            pickle.dump(self.model, fp)
        with open(model_path + '_scaler.pkl', 'wb') as fp:
            pickle.dump(self.scaler, fp)
        with open(model_path + '.json', 'w') as fp:
            json.dump(self.metadata, fp)

    def load(self, path: str):
        """Load model, scaler, and metadata."""
        model_path, _ = os.path.splitext(path)
        with open(model_path + '.pkl', 'rb') as fp:
            self.model = pickle.load(fp)
        with open(model_path + '_scaler.pkl', 'rb') as fp:
            self.scaler = pickle.load(fp)
        with open(model_path + '.json', 'r') as fp:
            self.metadata = json.load(fp)

    def _assess_tf_fraction(self, labels: pd.Series):
        """Throw an error for dramatically un-weighted data."""
        if labels.sum() > 0.8*len(labels):
            raise ValueError('Too many trues')
        elif labels.sum() < 0.2*len(labels):
            raise ValueError('Too many falses')

    def _create_logistic_regression(self):
        return sklearn.linear_model.LogisticRegression()

    def _create_random_forest(self):
        return sklearn.ensemble.RandomForestClassifier()

    def _create_xgboost(self):  
        # return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        return None

        

if __name__ == '__main__':
    import duq_ds3_2025.wine_quality
    wq = duq_ds3_2025.wine_quality.WineQuality()
    wq.read('data/wine+quality.zip')
    df = wq.df

    labels = df['quality']>5
    #features = df[['fixed acidity','sulfates','alcohol']] # [[]] turns into df not series
    features = df.drop(['quality'],axis=1)
    lr = ReusableClassifier('logistic_regression')
    lr.train(features,labels)

    rf = ReusableClassifier('random_forest') # works better because more complex
    rf.train(features,labels)

    lr.save('data/model_lr')
    lr.load('data/model_lr')

    lr.predict(features)