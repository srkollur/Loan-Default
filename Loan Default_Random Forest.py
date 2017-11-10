import quandl
import numpy as np
import scipy
import matplotlib.pyplot as plt
import xlsxwriter
import pandas as pd
import datetime
from datetime import datetime
import statistics
import pandas_datareader.data as data
import talib
import xlrd
import sklearn
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

df = pd.read_csv("train_v2.csv")

df.fillna(-99999, inplace=True)

X = np.array(df.drop(['loss'], 1))
Y = np.array(df['loss'])

X = preprocessing.scale(X)

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=10)

clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)



print(accuracy)

