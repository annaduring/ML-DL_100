# -*- coding: utf-8 -*-
"""
Spyder Editor

ML100 days - Day 1
Data Pre-Processing

"""

# import the libraries
import pandas as pd
import numpy as np

# read dataset & define in/dependent variables
df = pd.read_csv("datasets/Data.csv")

X = df.iloc[:, :3].values
y = df.iloc[:, -1:].values # redifined later as well 


# impute missing data
from sklearn.impute import SimpleImputer

imputing = SimpleImputer(missing_values=np.nan, strategy ="median", copy=True)

imputer = imputing.fit(X[:, 1:3])
# print(imputing.transform(X[:, 1:3]))
X[:, 1:3] = imputing.transform(X[:, 1:3])

# Encode Yes/No to Categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

le = LabelEncoder()

X[ : , 0] = le.fit_transform(X[ : , 0])

ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
le_Y = le
y =  le_Y.fit_transform(y)


# Split training and test 
from sklearn.model_selection  import train_test_split


X_train, X_test,  \
    y_train, y_test  = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)



