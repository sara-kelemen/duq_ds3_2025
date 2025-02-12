from sklearn.linear_model import LogisticRegression
import sklearn.linear_model
import sklearn.ensemble
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing 
import pickle 
import os
import json

class ReusableClassifier():
    def __init__(self,model_type: str):
        """Create a classifier storing a model and metadata.

        Args:
            model_type (str): can include random forests, log-reg, etc.
        """

        self.model_type = model_type
        if model_type == 'logistic_regression':
            self.model = self._create_logisitic_regression()
        elif model_type == 'random_forest':
            self.model = self._create_random_forest()
        else:
            raise ValueError("No model implemeneted of that type")
        
        # add all shared variables to the __init__
        self.scaler = None
        self.metadata = None

    def _create_logisitic_regression(self): # _ means private
        """Create a new logistic regression model from scikit-learn/sklearn."""
        return sklearn.linear_model.LogisticRegression()
    
    def _create_random_forest(self):
        """Create a new random forest model from scikit-learn/sklearn."""
        return sklearn.ensemble.RandomForestClassifier()
    
    def train(self, 
              features: pd.DataFrame, # multiple columns
              labels: pd.Series, # one dimensional array  (df column)
              test_frac: float = 0.1): #type hinting, unnecessary but good practice
        """Train the model from pandas data.

        Args:
            features (pd.DataFrame): input features, dataframe
            labels (pd.Series): input labels
            test_frac (float, optional): fraction of data to preserve for testing. Defaults to 0.1.
        """
        # big X is features, little y is for labels
        self._assess_tf_fraction(labels)

        # data scaling
        # 1. set min=0 max=1 but we need to consider outliers (scikit-learn min max scaler)
            # could remove outliers -- 2 standard deviations
        # 2. could use standard deviations to scale the data (standardize by standard normal)
            # standard scaler
        
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(features)
        features = self.scaler.transform(features)

        features_train, features_test, labels_train, labels_test = sklearn.model_selection.train_test_split(features, labels, test_size=test_frac)

        self.model.fit(features_train,labels_train)

        # assess performance of the model
        pred_labels = self.model.predict(features_test)
        accuracy = (pred_labels == labels_test).sum()/len(labels_test)
        accuracy = (pred_labels == labels_test).mean()

        self.metadata = {} # or else none error
        self.metadata['training_rows'] = len(features_train)
        self.metadata['accuracy'] = accuracy
        self.metadata['model_type'] = self.model_type
        print(f"Accuracy on test set is {accuracy}")

    def predict(self,features:pd.DataFrame):
        """Predict label from features."""
        self.scaler.transform(features)
        self.model.predict(features)

    def _assess_tf_fraction(self,labels:pd.Series):
        """Throw an error for dramatically unweighted data"""
        if labels.sum() > 0.8 * len(labels): #if >80% true
            raise ValueError("Too many Trues")
        elif labels.sum() < 0.2 * len(labels):
            raise ValueError("Too many Falses")
    
    def save(self,path:str):
        """Save model, scaler, and metadata to file"""
        model_path, ext = os.path.splitext(path)
        scaler_path = model_path + '_scaler.pkl'
        metadata_path = model_path + '.json'
        model_path = model_path + '.pkl'

        with open(model_path, 'wb') as fp: # wb means write binary/bytes
            pickle.dump(self.model, fp)
        with open(scaler_path, 'wb') as fp: 
            pickle.dump(self.scaler, fp)
        with open(metadata_path,'w') as fp:
            json.dump(self.metadata,fp)

    def load(self,path:str):
        """Load model, scaler, and metadata to file"""
        model_path, ext = os.path.splitext(path)
        scaler_path = model_path + '_scaler.pkl'
        metadata_path = model_path + '.json'
        model_path = model_path + '.pkl'

        with open(model_path, 'rb') as fp: 
            self.model = pickle.load(fp)
        with open(scaler_path, 'rb') as fp: 
            self.scaler = pickle.load(fp)
        with open(metadata_path,'r') as fp:
            self.metadata = json.load(fp)
        

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