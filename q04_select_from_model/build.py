# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

def select_from_model (df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    sel = SelectFromModel(RandomForestClassifier(random_state=9))
    sel.fit(X,y)
    return  X.columns.values[sel.get_support()].tolist()
