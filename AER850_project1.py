
"""
AER850 Project1
@author:  Molynn Chang 500921213 
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split

# Step 1
data = pd.read_csv('/Users/myatphonemolynn/Downloads/Project 1 Data.csv') #Read in data
df = pd.DataFrame(data) #Convert to dataframe
dfnew = df.to_numpy() #Convert to array

# Step 2
X, Y, Z, Step = dfnew.T #Splitting columns
summary_stats = data.describe() #Statistical analysis
print(f'Statistical analysis of the dataset\n{summary_stats}')
#Split into training and test sets
X_train, X_test, Y_train, Y_test, Z_train, Z_test, Step_train, Step_test = train_test_split(X, Y, Z, Step, test_size=0.2, random_state=0)

#Scatter plot of the train dataset
plt.scatter(Step_train, X_train, label='X')
plt.scatter(Step_train, Y_train, label='Y')
plt.scatter(Step_train, Z_train, label='Z')
plt.title('Coordinates vs. Steps')
plt.ylabel('Coordinates')
plt.xlabel('Steps')
plt.legend()
plt.show()

#Histogram of the train dataset
sns.displot(X_train)
plt.title('X coordinate')
plt.show()
sns.displot(Y_train)
plt.title('Y coordinate')
plt.show()
sns.displot(Z_train)
plt.title('Z coordinate')
plt.show()

# Step 3
#Correlation plot using Spearman Correlation
corr = df.corr(method='spearman')
sns.heatmap(corr, vmin=-1, vmax=1, annot=True)

#Step 4
from sklearn.tree import DecisionTreeClassifier
#Processing dataset to use for training
Xtrain = X_train.reshape(-1, 1)
Ytrain = Y_train.reshape(-1, 1)
Ztrain = Z_train.reshape(-1, 1)
features1 = np.column_stack((Xtrain,Ytrain,Ztrain))

#Decision tree model development
#Define decision tree model
dtree = DecisionTreeClassifier()
#Setup grid parameters
gridDT = [{'max_depth':[1, 5, 10, 15], 'min_samples_split':[2, 4, 6, 8], 'min_samples_leaf':[1, 3, 5, 7]}]
#Perform grid search
clfDT = GridSearchCV(estimator = dtree, param_grid = gridDT, scoring = 'f1_macro', cv = 5,)
#Training model
clfDT.fit(features1, Step_train)
#Decision tree result visualization
print(f'Decision Tree model\n{clfDT.best_estimator_}')

#Logistic regression model development
from sklearn import linear_model
#Define logistic regression model
logr = linear_model.LogisticRegression(max_iter=10000, solver='lbfgs')
#Setup grid parameters
gridLR = [{'multi_class':['multinomial', 'ovr'], 'C':[1, 10, 20 ,50]}]
#Perform grid search cross-validation for hyperparameters
clfLR = GridSearchCV(estimator = logr, param_grid = gridLR, scoring = 'f1_macro', cv = 5)
#Training model
clfLR.fit(features1, Step_train)
#Display logistic regression model result
print(f'Logistic Regression model\n{clfLR.best_estimator_}')

#Random forest model development
from sklearn.ensemble import RandomForestClassifier
#Define random forest model
rf = RandomForestClassifier()
#Setup grid parameters
param_grid = {'n_estimators': [5, 10, 15], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
#Perform grid search cross-validation for hyperparameters
clfRF = GridSearchCV(estimator=rf, param_grid=param_grid, scoring = 'f1_macro', cv=5)
#Training model
clfRF.fit(features1, Step_train)
#Display random forest model result
print(f'Random Forest model\n{clfRF.best_estimator_}')

#Random forest with RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
#Setup parameters and perform randomized search cross-validation
clfRRF = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, scoring = 'f1_macro', cv = 5)
#Training model
clfRRF.fit(features1, Step_train)
#Display random forest result (RandomnizedsearchCV)
print(f'Random forest model with RandomizedsearchCV\n{clfRRF.best_estimator_}')

#Step 5
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix
#Processing dataset for testing
Xtest = X_test.reshape(-1, 1)
Ytest = Y_test.reshape(-1, 1)
Ztest = Z_test.reshape(-1, 1)
testfeatures = np.column_stack((Xtest,Ytest,Ztest))

#Decision tree model testing
SteppredictDT = clfDT.predict(testfeatures)
#Calculate and display performances of the test results
print('Performance of Descision tree')
print('Precision: %f' % precision_score(Step_test, SteppredictDT, average='macro'))
print('Accuracy: %f' % accuracy_score(Step_test, SteppredictDT))
print('F1 Score: %f' % f1_score(Step_test, SteppredictDT, average='macro'))
#Generating confusion matrix
cmDT = confusion_matrix(Step_test, SteppredictDT)
#Plot confusion matrix
sns.heatmap(cmDT, cmap = 'Blues', cbar = False, annot = True)
plt.title('Confusion Matrix for Descision tree')
plt.show()

#Logistic Regression model testing
SteppredictLR = clfLR.predict(testfeatures)
#Calculate and display performances of the test results
print('Performance of Logistic Regression')
print('Precision: %f' % precision_score(Step_test, SteppredictLR, average='macro'))
print('Accuracy: %f' % accuracy_score(Step_test, SteppredictLR))
print('F1 Score: %f' % f1_score(Step_test, SteppredictLR, average='macro'))
#Generating confusion matrix
cmDT = confusion_matrix(Step_test, SteppredictLR)
#Plot confusion matrix
sns.heatmap(cmDT, cmap = 'Blues', cbar = False, annot = True)
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

#Random forest model testing
SteppredictRF = clfRF.predict(testfeatures)
#Calculate and display performances of the test results
print('Performance of Random forest')
print('Precision: %f' % precision_score(Step_test, SteppredictRF, average='macro'))
print('Accuracy: %f' % accuracy_score(Step_test, SteppredictRF))
print('F1 Score: %f' % f1_score(Step_test, SteppredictRF, average='macro'))
#Generating confusion matrix
cmRF = confusion_matrix(Step_test, SteppredictRF)
#Plot confusion matrix
sns.heatmap(cmRF, cmap = 'Blues', cbar = False, annot = True)
plt.title('Confusion Matrix for Random Forest')
plt.show()

#Random forest with RandomizedSearchCV model testing
SteppredictRRF = clfRRF.predict(testfeatures)
#Calculate and display performances of the test results
print('Performance of Random forest with RandomizedSearchCV')
print('Precision: %f' % precision_score(Step_test, SteppredictRRF, average='macro'))
print('Accuracy: %f' % accuracy_score(Step_test, SteppredictRRF))
print('F1 Score: %f' % f1_score(Step_test, SteppredictRRF, average='macro'))
#Generating confusion matrix
cmRRF = confusion_matrix(Step_test, SteppredictRRF)
#Plot confusion matrix
sns.heatmap(cmRRF, cmap = 'Blues', cbar = False, annot = True)
plt.title('Confusion Matrix for Random Forest with RandomizedSearchCV')
plt.show()

#Step 6
#Stacking models
from sklearn.ensemble import StackingClassifier
#Setup base models: Decision tree and Logistic Regression
#Parameters obtained from step 4
base_models = [('dt', clfDT.best_estimator_), ('lr', clfLR.best_estimator_)]
#Setup meta model
meta_model = RandomForestClassifier()
#Stack the models
clfStack = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
#Train the stacked model
clfStack.fit(features1, Step_train)
#Perform testing
SteppredictStack = clfStack.predict(testfeatures)
#Calculate and print out the performances
print('Performance of Stacking classifier')
print('Precision: %f' % precision_score(Step_test, SteppredictStack, average='macro'))
print('Accuracy: %f' % accuracy_score(Step_test, SteppredictStack))
print('F1 Score: %f' % f1_score(Step_test, SteppredictStack, average='macro'))
#Generate confusion matrix
cmST = confusion_matrix(Step_test, SteppredictStack)
#Plotting confusion matrix
sns.heatmap(cmST, cmap = 'Blues', cbar = False, annot = True)
plt.title('Confusion Matrix for Stacked classifier')
plt.show()

#Step 7
import joblib
#Package the stacked model
joblib.dump(clfStack,'stacking.joblib')
#Load the model
loaded_model = joblib.load('stacking.joblib')
#Setup coordinates to be predicted
coor = np.array([[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]])
#Predict from the provided coordinates
predicted_class = loaded_model.predict(coor)
#Print out the results
print('Predicted Steps')
for i, sample in enumerate(coor):
    print(f"Coordinates: {sample}, Step: {predicted_class[i]}")