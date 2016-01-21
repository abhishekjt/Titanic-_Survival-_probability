from pandas import *
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
from array import *
import math
titanic= pandas.read_csv('titanic_data.csv')
titanic=DataFrame(titanic)

'''in order to replace nan entries by median so that the probability will be 50-50'''
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

titanic.loc[titanic["Sex"] == "male", "Sex"]=0      #select all male sex and assign  as 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1 #female as 1
#  Find all the unique values for "Embarked".s=0,c=1,Q=2
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# Initialize our algorithm class
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
#  set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train,:])
    train_target = titanic["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

# The predictions are in three separate numpy arrays.  Concatenate them into one.
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
#calculate accuracy , try to map predictions to the data set (actually survived)
#accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions) This isn't working
x=0
sum=0
for i in range(0,len(predictions)):
    print(predictions[i],titanic["Survived"][i])
    if predictions[i]==titanic["Survived"][i]:
         sum=sum+predictions[i]


accuracy=sum/len(predictions)
print(accuracy)
