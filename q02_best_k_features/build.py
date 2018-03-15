# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


def percentile_k_features(df, k=20):
    X = df.iloc[:,:-1]
    y = df['SalePrice']
    lst=[]

    fs = SelectPercentile(f_regression, percentile=k)
    fs.fit_transform(X, y)

    col_nam =  X.columns.values[fs.get_support()]
    col_scr = fs.scores_[fs.get_support()]
    nam_scr = list(zip(col_nam,col_scr))

    srt_nam_scr = sorted(nam_scr, key=lambda x: x[1], reverse=True)
    for i in srt_nam_scr:
        lst.append(i[0])

    return lst
