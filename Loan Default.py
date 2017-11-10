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

##clf = RandomForestClassifier(n_estimators=10)
###clf3 = KNeighborsClassifier()
###clf4 = DecisionTreeClassifier()
###clf5 = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
###clf6 = SVC()
###clf7 = GaussianNB()

##clf.fit(X_train, Y_train)
##accuracy = clf.score(X_test, Y_test)

##clf3.fit(X_train, Y_train)
##accuracy3 = clf3.score(X_test, Y_test)

##clf4.fit(X_train, Y_train)
##accuracy4 = clf4.score(X_test, Y_test)

##clf5.fit(X_train, Y_train)
##accuracy5 = clf5.score(X_test, Y_test)

##clf6.fit(X_train, Y_train)
##accuracy6 = clf6.score(X_test, Y_test)

##clf7.fit(X_train, Y_train)
##accuracy7 = clf7.score(X_test, Y_test)

##print(accuracy)

##print(accuracy3)

##print(accuracy4)

##print(accuracy5)

##print(accuracy6)

##print(accuracy7)


#----------------------------------------------------------------------------

#clf1 = DecisionTreeClassifier()
##clf2 = KNeighborsClassifier(n_neighbors = 2, weights='distance',)
##clf3 = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
##clf1 = clf1.fit(X_train, Y_train)
##clf2 = clf2.fit(X_train, Y_train)
##clf3 = clf3.fit(X_train, Y_train)
##eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,2,1])
##eclf = eclf.fit(X_train, Y_train)

##accuracy = eclf.score(X_test, Y_test)

##print(accuracy)
#---------------------------------------------------------------------------
