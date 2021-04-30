# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('Dry_Bean_Dataset.xlsx')

#Seker, Barbunya, Bombay, Cali, Dermosan, Horoz and Sira
bean_seker = df[df['Class'] == 'SEKER']
bean_barbunya = df[df['Class'] == 'BARBUNYA']
bean_bombay = df[df['Class'] == 'BOMBAY']
bean_cali = df[df['Class'] == 'CALI']
bean_dermosan = df[df['Class'] == 'DERMOSAN']
bean_horoz = df[df['Class'] == 'HOROZ']
bean_sira = df[df['Class'] == 'SIRA']

# EquivDiameter
sns.FacetGrid(df, hue="Class", height=5) \
   .map(sns.histplot, "EquivDiameter") \
   .add_legend();
# Roundness
sns.FacetGrid(df, hue="Class", height=5) \
   .map(sns.histplot, "roundness") \
   .add_legend();
# Compactness
sns.FacetGrid(df, hue="Class", height=5) \
   .map(sns.histplot, "Compactness") \
   .add_legend();
# Perimeter
sns.FacetGrid(df, hue="Class", height=5) \
   .map(sns.histplot, "Perimeter") \
   .add_legend();
plt.show()

# +
data = df.to_numpy()
X = data[:, 0:16]
y = data[:, 16]

print(X.shape)
print('Class labels:', np.unique(y))

# +
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.28, random_state=None, stratify=y)
print(X_train.shape)
print(X_test.shape)
# -

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# ### Logistic Regression

# +
from sklearn.linear_model import LogisticRegression

#setting up the hyperparameter grid
param_grid = [{'C': np.logspace(-4, 2, 7)}]
lr = LogisticRegression()
#using gridsearch cross validation in order to find the best hyperparameters
gs = GridSearchCV(estimator=lr, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=10,
    n_jobs=-1)
gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)
lr = gs.best_estimator_

lr_t_acc = gs.best_score_
print('Accuracy training: ', lr_t_acc)
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)
lr_acc = accuracy_score(y_test, y_pred_test)

print('Accuracy test: ', lr_acc)


#the confusion matrix for training and testing data
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
confmat = confusion_matrix(y_test, y_pred_test)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('images/06_09.png', dpi=300)
plt.show()
# -

# ### Support Vector Machine

# +
from sklearn.svm import SVC

#setting the range for c and gamma
C_range = np.logspace(-4, 2, 7)
gamma_range = np.logspace(-4, 2, 7)

#setting up the parameters with their respective kernels
param_grid = [{'C': C_range, 'kernel': ['linear']},
    {'C': C_range, 
    'gamma': gamma_range, 
    'kernel': ['rbf']}]

svc = SVC()
#setting up the gridsearch cross validation
gs = GridSearchCV(estimator=svc, 
    param_grid=param_grid, 
    scoring='accuracy', 
    cv=10,
    n_jobs=-1)

gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)
svc = gs.best_estimator_

# Calculate the scores of training and testing data
svm_t_acc = gs.best_score_
print('Accuracy: ', svm_t_acc)

y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

svm_acc = accuracy_score(y_test, y_pred_test)

print('Accuracy test: ', svm_acc)

#confusion matrix for training and testing
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

# ### MLP Classifier

# +
#making the hidden layer range that is used
hls_range = [(8, 8), (10,10), (8,10)]
print(hls_range)
#making a range of alpha values for the model
alpha_range = np.logspace(-2,2,5)
print(alpha_range)
#creating the parameter grid
param_grid = [{'alpha':alpha_range, 'hidden_layer_sizes':hls_range}]

#setting up the model
gs = GridSearchCV(estimator=MLPClassifier(tol=1e-5, 
                                          learning_rate_init=0.02,
                                          max_iter=1000,
                                         random_state=1), 
                  param_grid=param_grid, 
                  cv=5)

gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)
mlp = gs.best_estimator_

#Retrain the data with the best estimater
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

gs.best_estimator_.fit(X_train_std, y_train)

mlp_t_acc =  gs.best_estimator_.score(X_train_std,y_train)
mlp_acc = gs.best_estimator_.score(X_test_std,y_test)

print("The accuracy for the training data is :", mlp_t_acc)
print("The accuracy for the test data is :", mlp_acc)

#confusion matricies
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

# ### Decision Tree

# +
from sklearn.tree import DecisionTreeClassifier

param_grid=[{'max_depth': [9, 10, 11, 12, 13, 14]}]

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)
dt = gs.best_estimator_

#Retrain the data with the best estimater
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

gs.best_estimator_.fit(X_train_std, y_train)

dt_t_acc = gs.best_estimator_.score(X_train_std,y_train)
dt_acc = gs.best_estimator_.score(X_test_std,y_test)

print("The accuracy for the training data is :", dt_t_acc)
print("The accuracy for the test data is :", dt_acc)

#confusion matricies
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

# ### Random Forest

# +
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

param_gird={
    'max_depth': [8, 9, 10],
}

gs = GridSearchCV(estimator=RandomForestClassifier(random_state=1),
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3)

gs.fit(X_train_std, y_train)
print(gs.best_estimator_)
print(gs.best_params_)
rfc = gs.best_estimator_

#Retrain the data with the best estimater
y_pred_train = gs.best_estimator_.predict(X_train_std)
y_pred_test = gs.best_estimator_.predict(X_test_std)

gs.best_estimator_.fit(X_train_std, y_train)

rf_t_acc = gs.best_estimator_.score(X_train_std,y_train)
rf_acc = gs.best_estimator_.score(X_test_std,y_test)

print("The accuracy for the training data is :", rf_t_acc)
print("The accuracy for the test data is :", rf_acc)

#confusion matricies
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_train, y_pred_train))
print(' TN, FN,\n FP, TP\n', confusion_matrix(y_test, y_pred_test))
# -

# ### Evaluating the Performance of the Model

# +
import matplotlib.pyplot as plt
# %matplotlib inline

models = ["LR", "SVM", "MLP", "DT", "RF"]
accuracy = [a*100 for a in [lr_acc, svm_acc, mlp_acc, dt_acc, rf_acc]]
incorrect = [100-a for a in accuracy]
xpos = np.arange(len(models))

plt.xticks(xpos, models)
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Performance of Classification Algorithms")
plt.bar(xpos, accuracy, label="Correctly classified instances")
plt.bar(xpos, incorrect, label="Incorrectly classified instances")
plt.legend(loc="best")
ax = plt.gca()

# +
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores =\
                learning_curve(estimator=lr,
                               X=X_train_std,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1., 10),
                               cv=10,
                               scoring='accuracy',
                               n_jobs=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi=300)
plt.show()

# +
from sklearn.model_selection import validation_curve


#param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
param_range = np.logspace(-3, 2, 11)
# param_range = [6,7,8, 9, 10, 11]
train_scores, test_scores = validation_curve(
                estimator=lr, 
                X=X_train_std, 
                y=y_train, 
                param_name= 'C',
                param_range = param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='green', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi=300)
plt.show()
# -


