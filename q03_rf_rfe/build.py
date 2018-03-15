# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


def rf_rfe(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    model =RandomForestClassifier()

    rfe = RFE(model)
    rfe = rfe.fit(X, y)

    return X.columns.values[rfe.get_support()].tolist()
