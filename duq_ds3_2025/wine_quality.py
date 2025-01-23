"""
    Example of classes that we already know.
    OOP (Object Oriented Programming): Programming with 'objects'. Objects are
    nothing other than a single copy of a class with specific data inside.
    The single copy, because we are annoying programmers, is called an 
    'instance' of the class.

    A class is a combined definition of some data and some specific functions.

    a = 'Duquesne'
    a.find('Duq)
    
    class String:
        # Init is the first function run and it is a dumb name that nobody would accidentally use.
        def __init__(self):
            # self is the name of the specific copy of the data/class
            # otherwise known as the instance of the class
            self.content = ''

        def set(new_content):
            self.content = new_content

        def find(self, substring):
            # look for substring
"""
import pandas as pd
import zipfile
import numpy as np

class WineQuality: # or WineQuality()
    def __init__(self):
        # add all class specific variables here
        self.df = None
        self.train_df =None
        self.test_df = None
    def read(self,path: str):
        """Read in and clean data"""
        zf = zipfile.ZipFile(path)
        self.df = pd.read_csv(zf.open('winequality-white.csv'),sep=';')

    def train(self,test_frac: float = 0.1):
        """Return only the training data. Identify the training data if it does not yet exist"""
        if self.train_df is None:
            self._train_test_split(test_frac)
        return self.train_df
    
    def test(self,test_frac: float = 0.1):
        """Return only the test data. Idenitfy test data if it DNE"""
        if self.test_df is None:
            self._train_test_split(test_frac)
        return self.test_df
    
    # if it starts with an underscore, officially you cannot reference it from another file
    # basically saying dont touch this
    def _train_test_split(self,test_frac: float):
        """Randomly train test split the df"""
        all_rows = np.arange(len(self.df))
        np.random.shuffle(all_rows)
        test_n_rows = round(len(self.df)*test_frac)
        test_rows = all_rows[:test_n_rows]
        train_rows = all_rows[test_n_rows:]

        self.train_df = self.df.loc[train_rows].reset_index(drop=True)
        self.test_df = self.df.loc[test_rows].reset_index(drop=True)

if __name__ == '__main__':
    wq = WineQuality() # parentheses means go
    wq.read=('data/wine+quality.zip')
    print(wq.test())