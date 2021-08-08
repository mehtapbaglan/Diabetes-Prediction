# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing as preprocessing
import seaborn as sns
from pyexpat import model
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
from sklearn.naive_bayes import GaussianNB


import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv('diabetes.csv')

# Preview data
print(dataset.head())

dataset.info()

# Statistical summary
print(dataset.describe().T)
#
# Count of null values
print(dataset.isnull().sum())


# Outcome countplot
sns.countplot(x = 'Outcome',data = dataset)

# Histogram of each feature

#
col = dataset.columns[:8]
plt.subplots(figsize = (20, 15))
length = len(col)
#
for i, j in itertools.zip_longest(col, range(length)):
    plt.subplot((length/2), 3, j + 1)
    plt.subplots_adjust(wspace = 0.1,hspace = 0.5)
    dataset[i].hist(bins = 20)
    plt.title(i)
plt.show()

# Scatter plot matrix

scatter_matrix(dataset, figsize = (20, 20));
#
# Pairplot
sns.pairplot(data = dataset, hue = 'Outcome')
plt.show()

# Heatmap
sns.heatmap(dataset.corr(), annot = True)
plt.show()


dataset_new = dataset

# Replacing zero values with NaN
dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)

# Count of NaN
print(dataset_new.isnull().sum())

#Replacing NaN with mean values
dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)




# Feature scaling using MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(dataset_new)
dataset_scaled = pd.DataFrame(dataset_scaled)
print(dataset_scaled)



# Selecting features - [Glucose, Insulin, BMI, Age]
X = dataset_scaled.iloc[:, [1, 4, 5, 7]].values
Y = dataset_scaled.iloc[:, 8].values

# Splitting X and Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42, stratify = dataset_new['Outcome'] )

# Checking dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# KNN Plotting a graph for n_neighbors
#
X_axis = list(range(1, 31))
acc = pd.Series()
x = range(1,31)
#
for i in list(range(1, 31)):
    knn_model = KNeighborsClassifier(n_neighbors = i)
    knn_model.fit(X_train, Y_train)
    prediction = knn_model.predict(X_test)
    acc = acc.append(pd.Series(metrics.accuracy_score(prediction, Y_test)))
plt.plot(X_axis, acc)
plt.xticks(x)
plt.title("Finding best value for n_estimators")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.grid()
plt.show()
print('Highest value: ',acc.values.max())
#
# K nearest neighbors Algorithm

knn = KNeighborsClassifier(n_neighbors = 17, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
#
Y_pred_knn = knn.predict(X_test)

accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
print("K Nearest neighbors accuracy without k-fold cv: " + str(accuracy_knn * 100))
#
# Kfold Split
folds = KFold(n_splits=10, shuffle=True, random_state=35)
knn_acc_scores = []
knn_prec_scores=[]
knn_recall_scores=[]
knn_rmse_scores=[]
knn_mse_scores=[]
knn_f1_scores=[]
#
for n_fold, (train_index, valid_index) in enumerate(folds.split(X, Y)):
    print('\n Fold ' + str(n_fold + 1) +
          ' \n\n train ids :' + str(train_index) +
          ' \n\n validation ids :' + str(valid_index))
#
    X_train, X_valid = X[train_index], X[valid_index]
    Y_train, Y_valid = Y[train_index], Y[valid_index]
#
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
#
    knn_acc_score = accuracy_score(Y_test, y_pred)
    knn_acc_scores.append(knn_acc_score)
    print('\n Accuracy score for Fold ' + str(n_fold + 1) + ' --> ' + str(knn_acc_score) + '\n')
#
    knn_prec_score=precision_score(Y_test, y_pred)
    knn_prec_scores.append(knn_prec_score)
    print('\n Precision score for Fold ' + str(n_fold + 1) + ' --> ' + str(knn_prec_score) + '\n')
#
    knn_rec_score = recall_score(Y_test, y_pred)
    knn_recall_scores.append(knn_rec_score)
    print('\n Recall score for Fold ' + str(n_fold + 1) + ' --> ' + str(knn_rec_score) + '\n')
#
    knn_f_score = f1_score(Y_test, y_pred)
    knn_f1_scores.append(knn_f_score)
    print('\n F1 score for Fold ' + str(n_fold + 1) + ' --> ' + str(knn_f_score) + '\n')
#
    knn_rmse_score = mean_squared_error(Y_test, y_pred, squared=True)
    knn_rmse_scores.append(knn_rmse_score)
    print('\n Rmse score for Fold ' + str(n_fold + 1) + ' --> ' + str(knn_rmse_score) + '\n')
#
    knn_mse_score = mean_squared_error(Y_test, y_pred, squared=False)
    knn_mse_scores.append(knn_mse_score)
    print('\n Mse score for Fold ' + str(n_fold + 1) + ' --> ' + str(knn_mse_score) + '\n')
print("\n Accurcy array: ")
print(knn_acc_scores)
print("\n Precision array: ")
print(knn_prec_scores)
print("\n Recall array: ")
print(knn_recall_scores)
print("\n F1 score array: ")
print(knn_f1_scores)
print("\n Rmse array: ")
print(knn_rmse_scores)
print("\n Mse array: ")
print(knn_mse_scores)

print('Avg. accuracy score :' + str(np.mean(knn_acc_scores)))
print('Avg. Precision score :' + str(np.mean(knn_prec_scores)))
print('Avg. Recall score :' + str(np.mean(knn_recall_scores)))
print('Avg. F1 score :' + str(np.mean(knn_f1_scores)))
print('Avg. Rmse score :' + str(np.mean(knn_rmse_scores)))
print('Avg. Mse score : \n' + str(np.mean(knn_mse_scores)))





#
# #####DECISION TREE
# function for fitting trees of various depths on the training data using cross-validation
def run_cross_validation_on_trees(X, Y, tree_depths, cv=10, scoring='accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = DecisionTreeClassifier(max_depth=depth)
        cv_scores = cross_val_score(tree_model, X, Y, cv=cv, scoring=scoring)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, Y).score(X, Y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores

# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()

# fitting trees of depth 1 to 24
sm_tree_depths = range(1,25)
sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, Y_train, sm_tree_depths)

# plotting accuracy
plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores,
                               'Accuracy per decision tree depth on training data')

plt.show()
#
idx_max = sm_cv_scores_mean.argmax()
sm_best_tree_depth = sm_tree_depths[idx_max]
sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]
sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]
print('The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(
      sm_best_tree_depth, round(sm_best_tree_cv_score*100,5), round(sm_best_tree_cv_score_std*100, 5)))
#
#
# function for training and evaluating a tree
#
dtc = DecisionTreeClassifier(max_depth=sm_best_tree_depth).fit(X_train, Y_train)
accuracy_train = dtc.score(X_train, Y_train)
accuracy_test = dtc.score(X_test, Y_test)
print('Single tree depth: ', sm_best_tree_depth)

#
Y_pred_dtc = dtc.predict(X_test)
accuracy_dtc = accuracy_score(Y_test, Y_pred_dtc)



# Kfold Split
folds = KFold(n_splits=10, shuffle=True, random_state=35)
dtc_acc_scores = []
dtc_prec_scores=[]
dtc_recall_scores=[]
dtc_rmse_scores=[]
dtc_mse_scores=[]
dtc_f1_scores=[]
#
for n_fold, (train_index, valid_index) in enumerate(folds.split(X, Y)):
    print('\n Fold ' + str(n_fold + 1) +
          ' \n\n train ids :' + str(train_index) +
          ' \n\n validation ids :' + str(valid_index))
    X_train, X_valid = X[train_index], X[valid_index]
    Y_train, Y_valid = Y[train_index], Y[valid_index]
    dtc.fit(X_train, Y_train)
    Y_pred = dtc.predict(X_test)
# #
    dtc_acc_score = accuracy_score(Y_test, Y_pred)
    dtc_acc_scores.append(dtc_acc_score)
    print('\n Accuracy score for Fold ' + str(n_fold + 1) + ' --> ' + str(dtc_acc_score) + '\n')
    #
    dtc_prec_score = precision_score(Y_test, Y_pred)
    dtc_prec_scores.append(dtc_prec_score)
    print('\n Precision score for Fold ' + str(n_fold + 1) + ' --> ' + str(dtc_prec_score) + '\n')
    #
    dtc_rec_score = recall_score(Y_test, Y_pred)
    dtc_recall_scores.append(dtc_rec_score)
    print('\n Recall score for Fold ' + str(n_fold + 1) + ' --> ' + str(dtc_rec_score) + '\n')
    #
    dtc_f_score = f1_score(Y_test, Y_pred)
    dtc_f1_scores.append(dtc_f_score)
    print('\n F1 score for Fold ' + str(n_fold + 1) + ' --> ' + str(dtc_f_score) + '\n')
    #
    dtc_rmse_score = mean_squared_error(Y_test, Y_pred, squared=True)
    dtc_rmse_scores.append(dtc_rmse_score)
    print('\n Rmse score for Fold ' + str(n_fold + 1) + ' --> ' + str(dtc_rmse_score) + '\n')
    #
    dtc_mse_score = mean_squared_error(Y_test, Y_pred, squared=False)
    dtc_mse_scores.append(dtc_mse_score)
    print('\n Mse score for Fold ' + str(n_fold + 1) + ' --> ' + str(dtc_mse_score) + '\n')
print("\n Accurcy array: ")
print(dtc_acc_scores)
print("\n Precision array: ")
print(dtc_prec_scores)
print("\n Recall array: ")
print(dtc_recall_scores)
print("\n F1 score array: ")
print(dtc_f1_scores)
print("\n Rmse array: ")
print(dtc_rmse_scores)
print("\n Mse array: ")
print(dtc_mse_scores)

print('Avg. accuracy score :' + str(np.mean(dtc_acc_scores)))
print('Avg. Precision score :' + str(np.mean(dtc_prec_scores)))
print('Avg. Recall score :' + str(np.mean(dtc_recall_scores)))
print('Avg. F1 score :' + str(np.mean(dtc_f1_scores)))
print('Avg. Rmse score :' + str(np.mean(dtc_rmse_scores)))
print('Avg. Mse score :' + str(np.mean(dtc_mse_scores)))
# #


text_representation = tree.export_text(dtc)
print(text_representation)


# Naive Bayes Algorithm

# We create a object from GaussianNB class
gnb = GaussianNB()

gnb.fit(X_train, Y_train)

result = gnb.predict(X_test)
accuracy_nb = accuracy_score(Y_test, result)
print("Naive Bayes: " + str(accuracy_nb * 100))


accuracy = accuracy_score(Y_test, result)

print(accuracy)

# Kfold Split
folds = KFold(n_splits=10, shuffle=True, random_state=35)
naive_bayes_acc_scores = []
naive_bayes_prec_scores=[]
naive_bayes_prec_recall_scores=[]
naive_bayes_prec_rmse_scores=[]
naive_bayes_prec_mse_scores=[]
naive_bayes_prec_f1_scores=[]

for n_fold, (train_index, valid_index) in enumerate(folds.split(X, Y)):
    print('\n Fold ' + str(n_fold + 1) +
          ' \n\n train ids :' + str(train_index) +
          ' \n\n validation ids :' + str(valid_index))
    X_train, X_valid = X[train_index], X[valid_index]
    Y_train, Y_valid = Y[train_index], Y[valid_index]
    gnb.fit(X_train, Y_train)
    Y_pred = gnb.predict(X_test)
#
    naive_bayes_acc_score = accuracy_score(Y_test, Y_pred)
    naive_bayes_acc_scores.append(naive_bayes_acc_score)
    print('\n Accuracy score for Fold ' + str(n_fold + 1) + ' --> ' + str(naive_bayes_acc_score) + '\n')
    naive_bayes_prec_score=precision_score(Y_test, Y_pred)
    naive_bayes_prec_scores.append(naive_bayes_prec_score)
    print('\n Precision score for Fold ' + str(n_fold + 1) + ' --> ' + str(naive_bayes_prec_score) + '\n')
#
    naive_bayes_prec_rec_score = recall_score(Y_test, Y_pred)
    naive_bayes_prec_recall_scores.append(naive_bayes_prec_rec_score)
    print('\n Recall score for Fold ' + str(n_fold + 1) + ' --> ' + str(naive_bayes_prec_rec_score) + '\n')
#
    naive_bayes_prec_f_score = f1_score(Y_test, Y_pred)
    naive_bayes_prec_f1_scores.append(naive_bayes_prec_f_score)
    print('\n F1 score for Fold ' + str(n_fold + 1) + ' --> ' + str(naive_bayes_prec_f_score) + '\n')
#
    naive_bayes_prec_rmse_score = mean_squared_error(Y_test, Y_pred, squared=True)
    naive_bayes_prec_rmse_scores.append(naive_bayes_prec_rmse_score)
    print('\n Rmse score for Fold ' + str(n_fold + 1) + ' --> ' + str(naive_bayes_prec_rmse_score) + '\n')
#
    naive_bayes_prec_mse_score = mean_squared_error(Y_test, Y_pred, squared=False)
    naive_bayes_prec_mse_scores.append(naive_bayes_prec_mse_score)
    print('\n Mse score for Fold ' + str(n_fold + 1) + ' --> ' + str(naive_bayes_prec_mse_score) + '\n')


print("\n Accurcy array: ")
print(naive_bayes_acc_scores)
print("\n Precision array: ")
print(naive_bayes_prec_scores)
print("\n Recall array: ")
print(naive_bayes_prec_recall_scores)
print("\n F1 score array: ")
print(naive_bayes_prec_f1_scores)
print("\n Rmse array: ")
print(naive_bayes_prec_rmse_scores)
print("\n Mse array: ")
print(naive_bayes_prec_mse_scores)

print('Avg. accuracy score :' + str(np.mean(naive_bayes_acc_scores)))
print('Avg. Precision score :' + str(np.mean(naive_bayes_prec_scores)))
print('Avg. Recall score :' + str(np.mean(naive_bayes_prec_recall_scores)))
print('Avg. Rmse score :' + str(np.mean(naive_bayes_prec_rmse_scores)))
print('Avg. Mse score :' + str(np.mean(naive_bayes_prec_mse_scores)))
#





