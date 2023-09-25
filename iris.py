import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

iris = pd.read_csv("Iris.csv") #load the dataset

iris.drop('Id',axis=1,inplace=True)

train, test = train_test_split(iris, test_size = 0.3)

train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]# taking the training data features
train_y=train.Species# output of our training data
test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # taking test data features
test_y =test.Species   #output value of test data

svm_model = svm.SVC() #select the algorithm
svm_model.fit(train_X,train_y) # we train the algorithm with the training data and the training output
prediction_svm = svm_model.predict(test_X) #now we pass the testing data to the trained algorithm
predict_X_svm = svm_model.predict(train_X)

lr = LogisticRegression()
lr.fit(train_X,train_y)
prediction_lr = lr.predict(test_X)
predict_X_lr = lr.predict(train_X)

dt=DecisionTreeClassifier()
dt.fit(train_X,train_y)
prediction_dt = dt.predict(test_X)
predict_X_dt = dt.predict(train_X)

kn=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class
kn.fit(train_X,train_y)
prediction_kn = kn.predict(test_X)
predict_X_kn = kn.predict(train_X)

# Calculate accuracy scores
accuracy_scores = {
    'SVM': metrics.accuracy_score(prediction_svm, test_y),
    'Logistic Regression': metrics.accuracy_score(prediction_lr, test_y),
    'Decision Tree': metrics.accuracy_score(prediction_dt, test_y),
    'KNN': metrics.accuracy_score(prediction_kn, test_y)
}

# Save accuracy scores to a text file
with open('accuracy_scores.txt', 'w') as file:
    for model, score in accuracy_scores.items():
        file.write(f'{model}: {score}\n')

