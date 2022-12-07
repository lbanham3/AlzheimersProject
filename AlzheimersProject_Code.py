# Alzheimer's fMRI Brain Scan Classification Project
# Author: Laura Banham
# Date Started: 11/03/2022

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

# Read in the MRI dataset
pathTrain = '/Users/laurabanham/OneDrive - Georgia Institute of Technology/CDA/Course Project/Alzheimer_s Dataset/train'
categories = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]

# Find out how many files there are for each category
numFiles = list()
for i in categories:
    fileName = pathTrain + "/" + i
    numFiles.append(len(os.listdir(fileName)))
numFiles

# Loop through and pull all of the images into a labeled dataframe
m=208
n=176
vectorLength = m*n
imgList = list()
for i in range(0, 2561):
    print("Round "+str(i))
    for j in categories:
        fileName = pathTrain + "/" + j + "/" + j.split("Demented")[0].lower() + "Dem" + str(i) + ".jpg"
        if os.path.exists(fileName):
            imgData = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            imgDataVector = imgData.reshape(vectorLength,)
            imgDataVector = imgDataVector.astype('float64')
            imgList.append([j.split("Demented")[0], imgDataVector])

alzArr = np.asarray(imgList, dtype=object)

# Grouping the dataset into sub-datasets by label
def group_dataset(arr):
    alzGrouped = [arr[arr[:,0]==k] for k in np.unique(arr[:,0])]
    categoriesOrdered = np.unique(arr[:,0])
    return(alzGrouped, categoriesOrdered)


# Splitting dataset into train:test subsets-------------------
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(np.vstack(alzArr[:,1]), alzArr[:,0], test_size=0.2, random_state=96)

unique, counts = np.unique(train_y, return_counts=True)
print("Labels: " + str(unique) + "\n Training set counts: " + str(counts))
unique2, counts2 = np.unique(test_y, return_counts=True)
print("Labels: " + str(unique2) + "\n Testing set counts: " + str(counts2))

# Fitting Models-----------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Finding the optimal parameters for each model--------------
# SVM
params = dict(C = [0.1, 1, 10, 100, 1000],
                gamma = [1, 0.1, 0.01, 0.001],
                kernel = ['linear', 'rbf'])
svm_grid = GridSearchCV(
    estimator=SVC(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1
)
svm_grid.fit(train_x, train_y)
print("SVM Best Parameters:", svm_grid.best_params_)
# SVM Best Parameters: {'C': 0.1, 'gamma': 1, 'kernel': 'linear'}

# KNN
k_range = list(range(2,100))
weight_options = ["uniform", "distance"]
params = dict(n_neighbors = k_range, weights = weight_options)
knn_grid = GridSearchCV(KNeighborsClassifier(), params, cv = 10, scoring = 'accuracy')
knn_grid.fit(train_x, train_y)
print("KNN Best Parameters:", knn_grid.best_params_)
## KNN Best Parameters: {'n_neighbors': 2, 'weights': 'distance'}

# Naive Bayes
params = dict(alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000])
nb_grid = GridSearchCV(
    estimator=MultinomialNB(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1
)
nb_grid.fit(train_x, train_y)
print("Naive Bayes Best Parameters:", nb_grid.best_params_)
# Naive Bayes Best Parameters: {'alpha': 0.1}

# Decision Tree
params = dict(criterion = ['gini', 'entropy', 'log_loss'])
dt_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1
)
dt_grid.fit(train_x, train_y)
print("Decision Tree Best Parameters:", dt_grid.best_params_)
# Decision Tree Best Parameters: {'criterion': 'entropy'}

# Random Forest
params = dict(n_estimators = [10, 50, 100, 150, 200],
                criterion = ['gini', 'entropy', 'log_loss'])
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1
)
rf_grid.fit(train_x, train_y)
print("Random Forest Best Parameters:", rf_grid.best_params_)
# Random Forest Best Parameters: {'criterion': 'gini', 'n_estimators': 150}

# Fitting the models with the optimal parameters-----------------------------
# Multinomial Logistic Regression------------
from sklearn import linear_model
from sklearn import metrics
multiLogReg = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)
multiLogReg_pred_y = multiLogReg.predict(test_x)
pd.crosstab(test_y, multiLogReg_pred_y)

print("Multinomial Logistic Regression Train Dataset Accuracy:", metrics.accuracy_score(train_y, multiLogReg.predict(train_x)))
print("Multinomial Logistic Regression Test Dataset Accuracy:", metrics.accuracy_score(test_y, multiLogReg_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, multiLogReg_pred_y))

# Support Vector Machine--------------
# SVM Best Parameters: {'C': 0.1, 'gamma': 1, 'kernel': 'linear'}
## Building and fitting the classifier
svmOptimal = SVC(kernel='linear', gamma=1, C=0.1).fit(train_x, train_y)

## Make predictions
# Predicting the test set results
svm_pred_y = svmOptimal.predict(test_x)

## Make confusion matrix
print(pd.crosstab(test_y, svm_pred_y))

## Print the accuracy of the model
print("SVM Train Dataset Accuracy:", metrics.accuracy_score(train_y, svmOptimal.predict(train_x)))
print("SVM Test Dataset Accuracy:", metrics.accuracy_score(test_y, svm_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, svm_pred_y))


# K Nearest Neighbors-------
## KNN Best Parameters: {'n_neighbors': 2, 'weights': 'distance'}
knnOptimal = KNeighborsClassifier(n_neighbors=2, weights='distance').fit(train_x, train_y)

## Make predictions
# Predicting the test set results
knn_pred_y = knnOptimal.predict(test_x)

## Make confusion matrix
print(pd.crosstab(test_y, knn_pred_y))

## Print the accuracy of the model
print("KNN Train Dataset Accuracy:", metrics.accuracy_score(train_y, knnOptimal.predict(train_x)))
print("KNN Test Dataset Accuracy:", metrics.accuracy_score(test_y, knn_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, knn_pred_y))

# Naive Bayes---------------
# Naive Bayes Best Parameters: {'alpha': 0.1}
nbOptimal = MultinomialNB(alpha=0.1).fit(train_x, train_y)

## Make predictions
# Predicting the test set results
nb_pred_y = nbOptimal.predict(test_x)

## Make confusion matrix
print(pd.crosstab(test_y, nb_pred_y))

## Print the accuracy of the model
print("Naive Bayes Train Dataset Accuracy:", metrics.accuracy_score(train_y, nbOptimal.predict(train_x)))
print("Naive Bayes Test Dataset Accuracy:", metrics.accuracy_score(test_y, nb_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, nb_pred_y))

# Decision Tree---------------
# Decision Tree Best Parameters: {'criterion': 'entropy'}
dtOptimal = DecisionTreeClassifier(criterion='entropy').fit(train_x, train_y)

## Make predictions
# Predicting the test set results
dt_pred_y = dtOptimal.predict(test_x) 

## Make confusion matrix
print(pd.crosstab(test_y, dt_pred_y))

## Print the accuracy of the model
print("Decision Tree Train Dataset Accuracy:", metrics.accuracy_score(train_y, dtOptimal.predict(train_x)))
print("Decision Tree Test Dataset Accuracy:", metrics.accuracy_score(test_y, dt_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, dt_pred_y))

# Random Forest----------
# Random Forest Best Parameters: {'criterion': 'gini', 'n_estimators': 150}
rfOptimal = RandomForestClassifier(criterion='gini', n_estimators=150).fit(train_x, train_y)

## Make predictions
# Predicting the test set results
rf_pred_y = rfOptimal.predict(test_x)

## Make confusion matrix
print(pd.crosstab(test_y, rf_pred_y))

## Print the accuracy of the model
print("Random Forest Train Dataset Accuracy:", metrics.accuracy_score(train_y, rfOptimal.predict(train_x)))
print("Random Forest Test Dataset Accuracy:", metrics.accuracy_score(test_y, rf_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, rf_pred_y))

# AdaBoost-------
params = dict(n_estimators = [10, 50, 100, 150, 200],
                base_estimator=[multiLogReg, svmOptimal, knnOptimal, nbOptimal, dtOptimal, rfOptimal])

adaB_grid = GridSearchCV(
    estimator=AdaBoostClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1
)
adaB_grid.fit(train_x, train_y)
print("AdaBoost Best Parameters:", adaB_grid.best_params_)
# AdaBoost Best Parameters: {'base_estimator': LogisticRegression(multi_class='multinomial', solver='newton-cg'), 'n_estimators': 10}
adaBOptimal = AdaBoostClassifier(n_estimators = 10, base_estimator = multiLogReg)
adaBOptimal.fit(train_x, train_y)

## Make predictions
# Predicting the test set results
adaB_pred_y = adaBOptimal.predict(test_x)

## Make confusion matrix
print(pd.crosstab(test_y, adaB_pred_y))

## Print the accuracy of the model
print("AdaBoost Train Dataset Accuracy:", metrics.accuracy_score(train_y, adaBOptimal.predict(train_x)))
print("AdaBoost Test Dataset Accuracy:", metrics.accuracy_score(test_y, adaB_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, adaB_pred_y))

# NEW MODEL - AdaBoost 2 -------------
from sklearn.ensemble import VotingClassifier
newMod = VotingClassifier([('svm', SVC(kernel='linear', gamma=1, C=0.1, probability=True)), 
                            ('logReg', linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg'))], voting='soft')
adaBoost2 = AdaBoostClassifier(base_estimator = newMod)
adaBoost2.fit(train_x, train_y)

## Make predictions
# Predicting the test set results
ab2_pred_y = adaBoost2.predict(test_x)

## Make confusion matrix
print(pd.crosstab(test_y, ab2_pred_y))

## Print the accuracy of the model
print("AdaBoost2 Train Dataset Accuracy:", metrics.accuracy_score(train_y, adaBoost2.predict(train_x)))
print("AdaBoost2 Test Dataset Accuracy:", metrics.accuracy_score(test_y, ab2_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, ab2_pred_y))


# Miscellaneous Code -------------------------------------

# Show average and standard deviation images
alzGrouped, catOrder = group_dataset(alzArr)
avgImgs = [0]*4
row = 1
col = 4
fig1 = plt.figure(figsize=(5, 5))
for i, groupedDataset in zip([0,1,2,3], alzGrouped):
    fig1.add_subplot(row, col, i+1)
    fig1.suptitle('Average Images')
    labelAvg = groupedDataset[0,0] + " Demented"
    avgImg = np.mean(groupedDataset[:,1], axis = 0)
    avgImgNp = np.matrix(avgImg)
    avgImgReshape = avgImgNp.reshape(m,n)
    plt.imshow(avgImgReshape, cmap="gray")
    plt.title(labelAvg, size=9)
    plt.axis('off')
    avgImgs[i] = avgImg
plt.show()

fig2 = plt.figure(figsize=(10, 7))
for i, groupedDataset in zip([0,1,2,3], alzGrouped):
    labelStdDev = groupedDataset[0,0] + " Demented"
    stdImg = np.std(groupedDataset[:,1], axis = 0)
    fig2.add_subplot(row, col, i+1)
    fig2.suptitle('Standard Deviation Images')
    stdImgNp = np.matrix(stdImg)
    stdImgReshape = stdImgNp.reshape(m,n)
    plt.imshow(stdImgReshape, cmap="gray")
    plt.title(labelStdDev, size=9)
    plt.axis('off')
plt.show()

# Plotting the difference between non- and moderately demented patients
# Note: Darker colors indicate differences
diff = 255.0 - cv2.absdiff(np.matrix(avgImgs[0]).reshape(m,n), np.matrix(avgImgs[3]).reshape(m,n))
plt.imshow(diff, 'gray')
plt.xticks(color='w')
plt.yticks(color='w')
plt.tick_params(left=False,bottom=False)
plt.title("Differences Between \n Averaged Moderate and Non-Demented Images")
plt.show()

# Plotting model results
import seaborn as sns
df = pd.DataFrame(data=np.array([['Logistic Regression', 0.9824, 0.98, 0.98, 0.98],
                                ['SVM',	0.9844,	0.98, 0.98,	0.98],
                                ['KNN',	0.9980,	1.00, 1.00, 1.00],
                                ['Naive Bayes', 0.4829,	0.51, 0.48,	0.48],
                                ['Decision Tree', 0.7278, 0.72,	0.73, 0.72],
                                ['Random Forest', 0.9083, 0.91, 0.91, 0.91],
                                ['AdaBoost 1', 0.9805, 0.98, 0.98, 0.98],
                                ['AdaBoost 2', 0.9844, 0.98, 0.98, 0.98]]), 
                                columns=['Model', 'Test Acc', 'Precision', 'Recall', 'F1'])

df2 = pd.melt(df, id_vars=['Model'], value_vars=["Test Acc", "Precision", "Recall", "F1"])
df2['value'] = df2['value'].astype('float')
# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df2, kind="bar",
    x="variable", y="value", hue="Model",
    alpha=1, height=6
)
g.despine(left=True)
g.set_axis_labels("Metric", "")
plt.show()
