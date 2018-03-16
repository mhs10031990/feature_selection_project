# Default imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()

def forward_selected(data, model):
    X=data.iloc[:,:-1]
    y=data.iloc[:,-1]

    X_col = X.columns
    X_col_itr = len(X_col) - 10

    #Initialize all the intermediate variables
    fnl_col=[]
    prev_score = -1.00
    final_score = - 1.00
    club_score = []
    select=[]

# Should execute till First 10 feature are selected having max r2_score
    while len(X_col) != X_col_itr:

            #Loop to scan each feature by fit on X,y and predicting on X again.
            for i in X_col:
                # Select one feature at a time
                sel = fnl_col + [i]

                # Fit the model for selected feature on X and y.
                model.fit(X[sel], y)
                y_pred = model.predict(X[sel])

                #Calculate the r2_score for each output
                score = r2_score(y, y_pred)

                #Logic to compare the r2_score with previous value (previous feature r2_score)
                if score > prev_score:
                    sel_col = i
                    prev_score = score

            # Logic to select r2_score, feature after each for loop execution is done.
            if prev_score > final_score:
                final_score = prev_score
                club_score.append(final_score)
                fnl_col.append(sel_col)

                # Remove the selected column from X_col.
                X_col = [a for a in X_col if a != sel_col]

    return fnl_col, club_score 
