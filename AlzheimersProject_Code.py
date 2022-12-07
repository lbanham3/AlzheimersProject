# Course Project
# Course: CDA
# Date: 11/03/2022

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

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


# Making dataset sparse CSR, splitting into train:test and scaling fitting models-------------------
# Making dataset into sparse CSR matrix
# import scipy
# ft_cols = [x for x in alzDf.columns if x != 'Label']
# sparse_x = scipy.sparse.csr_matrix(alzDf[ft_cols].astype(float).values)

# Splitting into train:test
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(np.vstack(alzArr[:,1]), alzArr[:,0], test_size=0.2, random_state=96)
unique, counts = np.unique(train_y, return_counts=True)
print("Labels: " + str(unique) + "\n Training set counts: " + str(counts))

unique2, counts2 = np.unique(test_y, return_counts=True)
print("Labels: " + str(unique2) + "\n Testing set counts: " + str(counts2))

train_x_sparse, test_x_sparse, train_y_sparse, test_y_sparse = train_test_split(sparse_x, alzDf['Label'], test_size=0.2, random_state=96)

# Scaling
# from sklearn.preprocessing import MinMaxScaler 
# scaler = MinMaxScaler()
# train_x_scale = scaler.fit_transform(train_x)
# test_x_scale = scaler.transform(test_x)

# Fitting Models-----------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


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

# Gradient Boosting
params = dict(n_estimators = [10, 50, 100, 150, 200])

gradB_grid = GridSearchCV(
    estimator=GradientBoostingClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1
)
gradB_grid.fit(train_x, train_y)
print("Gradient Boosting Best Parameters:", gradB_grid.best_params_)

# Neural Network
params = dict(hidden_layer_sizes = [(100, 75, 50), (10, 10, 10), (75, 50), (50, 50), (10, 10)],
              activation = ['logistic', 'tanh', 'relu'],
              alpha = [0.0001, 0.001, 0.005],
              early_stopping = [True, False])

nn_grid = GridSearchCV(
    estimator=MLPClassifier(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1
)
nn_grid.fit(train_x, train_y)
print("Neural Network Best Parameters:", nn_grid.best_params_)


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

# Gradient Boosting-------
# Gradient Boosting Best Parameters: {'n_estimators': 10}
gradBOptimal = GradientBoostingClassifier()
gradBOptimal.fit(train_x, train_y)


gradBOptimal = XGBClassifier()
d = {x: y for y, x in enumerate(set(train_y))}
train_y2 = [d[y] for y in train_y]
train_y2

d = {x: y for y, x in enumerate(set(test_y))}
test_y2 = [d[y] for y in test_y]
test_y2

gradBOptimal.fit(train_x, train_y2)


## Make predictions
# Predicting the test set results
gradB_pred_y = gradBOptimal.predict(test_x)

## Make confusion matrix
print(pd.crosstab(test_y2, gradB_pred_y))

## Print the accuracy of the model
print("Gradient Boosting Train Dataset Accuracy:", metrics.accuracy_score(train_y, gradBOptimal.predict(train_x)))
print("Gradient Boosting Test Dataset Accuracy:", metrics.accuracy_score(test_y, gradB_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, gradB_pred_y))

# Neural Network-------
# Neural Network Best Parameters: {'n_estimators': 10}
nnOptimal = MLPClassifier()
nnOptimal.fit(train_x, train_y)

## Make predictions
# Predicting the test set results
nn_pred_y = nnOptimal.predict(test_x)

## Make confusion matrix
print(pd.crosstab(test_y, nn_pred_y))

## Print the accuracy of the model
print("Neural Network Train Dataset Accuracy:", metrics.accuracy_score(train_y, nnOptimal.predict(train_x)))
print("Neural Network Test Dataset Accuracy:", metrics.accuracy_score(test_y, nn_pred_y))

# Precision, Recall and F1 Scores
print(metrics.classification_report(test_y, nn_pred_y))


# NEW MODEL
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


# -------------------------------------
# Show all images in dataframe
for i, row in alzDf.iterrows():
    img = row['Img'].reshape(m,n)
    label = row['Label']+str(i)
    plt.imshow(img, cmap="gray")
    plt.title(label)
    plt.show()

# Show one image (by index in dataframe)
img = alzDf['Img'][1].reshape(m,n)
label = alzDf['Label'][1]+str(1)
plt.imshow(img, cmap="gray")
plt.title(label)
plt.show()

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
df = pd.DataFrame(data=np.array([['Logistic Regression',	0.9824,	0.98, 0.98, 0.98],
                                ['SVM',	0.9844,	0.98,	0.98,	0.98],
                                ['KNN',	0.9980,	1.00,	1.00,	1.00],
                                ['Naive Bayes', 0.4829,	0.51,	0.48,	0.48],
                                ['Decision Tree', 0.7278,	0.72,	0.73,	0.72],
                                ['Random Forest',	0.9083,	0.91,	0.91,	0.91],
                                ['AdaBoost 1',	0.9805,	0.98,	0.98,	0.98],
                                ['AdaBoost 2', 	0.9844,	0.98,	0.98,	0.98]]), 
                                columns=['Model', "Test Acc", "Precision", "Recall", "F1"])

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

# PCA the dataset -------------------------
# Finding an optimal number of components using percent variance explained
x = alzDf.drop('Label', axis=1)
pca = PCA(n_components=200)
covMat = pca.fit(x)

variance = covMat.explained_variance_ratio_
var = np.cumsum(np.round(covMat.explained_variance_ratio_, decimals=5)*100)

plt.plot(var)
plt.ylabel("Percentage of Variance Explained")
plt.xlabel("Number of Features")
plt.title("PCA with 200 Components")
plt.show()

# ~90% of the variance is explained by 142 of the components

# Top 2 principal components
pca = PCA(n_components=2)
princeComp = pca.fit_transform(x)
principalDf = pd.DataFrame(princeComp, columns = ['principal component 1', 'principal component 2'])

# Plotting 2 Component PCA
fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('PCA for Top 2 Components', fontsize = 20)
colors = ['r', 'g', 'b', 'y']
for category, color in zip(categories, colors):
    indicesToKeep = alzDf['Label'] == category.split("Demented")[0]
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1'],
               principalDf.loc[indicesToKeep, 'principal component 2'],
               c = color,
               s = 50)
ax.legend(categories)
ax.grid()
plt.show()

# Plotting reconstructed image
twoDimData = pca.fit_transform(x)
newImage = pca.inverse_transform(twoDimData)
newImage[0].shape
plt.imshow(newImage[1].reshape(m,n),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
plt.show()


# PCA the dataset by GROUP -------------------------
alzGrouped, catOrder = group_dataset(alzArr, "Label")

def grouped_pca_var_explained(dfGrouped):
    var = list()
    for dataset in dfGrouped:
        xCat = dataset.drop('Label', axis=1)
        pcaCat = PCA(n_components=50)
        covMatCat = pcaCat.fit(xCat)
        varCat = np.cumsum(np.round(covMatCat.explained_variance_ratio_, decimals=5)*100)
        var.append(varCat)
    return(var)

var = grouped_pca_var_explained(alzGrouped)

for varSet, category in zip(var, catOrder):
    plt.plot(varSet)
    plt.ylabel("Percentage of Variance Explained")
    plt.xlabel("Number of Features")
    title1 = "PCA with 50 Components for " + category + " Demented"
    plt.title(title1)
    plt.show()

def grouped_pca_n_comp(nComp, dfGrouped, categoriesOrdered):
    pcDfCat = pd.DataFrame()
    newImage = list()
    colNames = list()
    for dataset, category in zip(dfGrouped, categoriesOrdered):
        xCat = dataset.drop('Label', axis=1)
        pcaCat = PCA(n_components=nComp)
        princeCompCat = pcaCat.fit_transform(xCat)
        newImage.append(pcaCat.inverse_transform(princeCompCat))
        for i in range(nComp):
            colNames.append("pc" + str(i%nComp+1) + category)
        principalDfCat = pd.DataFrame(princeCompCat)
        pcDfCat = pd.concat([pcDfCat, principalDfCat], axis=1)
    pcDfCat.columns = colNames
    return(pcDfCat, newImage)

groupPCDf, newImageList = grouped_pca_n_comp(20, alzGrouped, catOrder)

# Plotting 2 Component PCA by Group
for i, category in zip(range(0,8,2), catOrder):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 10)
    ax.set_ylabel('Principal Component 2', fontsize = 10)
    title2 = 'PCA for Top 2 Components of ' + str(category) + " Demented Patients"
    ax.set_title(title2, fontsize = 15)
    ax.scatter(groupPCDf.iloc[:,i],
                groupPCDf.iloc[:,i+1])
    ax.grid()
    plt.show()

# Plotting Reconstructed Image by Group
nComp = 20
for image, category in zip(newImageList, catOrder):
    plt.imshow(image[1].reshape(m,n),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255))
    title3 = 'Reconstructed Image with ' + str(nComp) + ' Components for ' + str(category) + " Demented Patients"
    plt.title(title3, fontsize = 10)
    plt.show()
