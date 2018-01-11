import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, \
    gaussian_process,model_selection,metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from operator import itemgetter
from sklearn.feature_selection import RFECV
import time
import operator
from sklearn.externals import joblib

def name_process(data):
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    data['Title'] = data['Title'].replace(['Sir', 'Countess', 'Lady', 'Dona'], 'Noble')
    data['Title'] = data['Title'].replace(['Mlle'], 'Miss')
    data['Title'] = data['Title'].replace(['Ms'], 'Miss')
    data['Title'] = data['Title'].replace(['Mme'], 'Mrs')
    data['Title'] = data['Title'].replace(['Col', 'Major', 'Dr'], 'Officer')
    data['Title'] = data['Title'].replace(['Mr', 'Jonkheer', 'Don', 'Rev', 'Capt'], 'Servant')
    title_map = {'Noble': 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'Officer': 5, 'Servant':6}
    data['Title'] = data['Title'].map(title_map)

    data['NameLength'] = data.Name.apply(lambda x: len(x))

def fare_process(data):
    for i in data["Pclass"].drop_duplicates():
        for j in data["Embarked"].drop_duplicates():
            data.loc[(data.Fare.isnull()) & (data.Pclass == i) & (data.Embarked == j), 'Fare'] \
                = data[(data['Pclass'] == i) & (data.Embarked == j)]['Fare'].dropna().median()


def get_person(data):
    data.loc[(data.Age < 14), 'Person'] = 'Child'
    data.loc[(data.Age >= 14) & (data.Sex == 'male'), 'Person'] = 'male_adult'
    data.loc[(data.Age >= 14) & (data.Sex == 'female'), 'Person'] = 'female_adult'
    dummies = pd.get_dummies(data['Person'])
    data = pd.concat([data, dummies], axis=1)
    data.drop(['Person'], axis=1, inplace=True)
    return data


def surname_process(nm):
    return nm.split(',')[0].lower()



def perishing_female_process(data):
    train = data[:891]
    perishing_female_surnames = list(set(train[(train.female_adult == 1) & (train.Survived == 0) &
                                              (train.Family_Size > 1)]['Surname'].values))
    data['Perishing_Mother'] = data.Surname.apply(lambda x:1 if x in perishing_female_surnames else 0)


def surviving_father_process(data):
    train = data[:891]
    surviving_male_surnames = list(set(train[(train.male_adult == 1) & (train.Survived == 1) &
                                                (train.Family_Size > 1)]['Surname'].values))
    data['Surviving_Father'] = data.Surname.apply(lambda x:1 if x in surviving_male_surnames else 0)


def fill_missing_age(data):
    for i in range(1, 4):
        median_age = data[(data["Title"] == i) & data["PassengerId"] <= 891]["Age"].median()
        data.loc[(data.Age.isnull()) & (data["PassengerId"] <= 891), 'Age'] = median_age

        median_age = data[(data["Title"] == i) & data["PassengerId"] <= 891]["Age"].median()
        data.loc[(data.Age.isnull()) & (data["PassengerId"] > 891), 'Age'] = median_age

    return data


if __name__=='__main__':
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    test["Survived"] = 0
    data = train.append(test)
    del train, test

    #Name
    name_process(data)

    #Embarked
    data['Embarked'].fillna('S', inplace=True)
    data['Embarked'] = pd.factorize(data['Embarked'])[0]

    #SibSp & Parch
    data['Family_Size'] = data['Parch'] + data['SibSp'] + 1

    #Fare
    fare_process(data)

    data = get_person(data)

    # Sex
    dummies = pd.get_dummies(data['Sex'], prefix='Sex')
    data = pd.concat([data, dummies], axis=1)

    data['Surname'] = data['Name'].apply(surname_process)
    perishing_female_process(data)
    surviving_father_process(data)

    #Age
    fill_missing_age(data)

    #Cabin
    data['Cabin'].fillna('U', inplace=True)
    data['Cabin'] = pd.factorize(data['Cabin'])[0]


    train = data[:891]
    test = data[891:]
    x = train.drop(['PassengerId', 'Name', 'Surname', 'Ticket', 'Sex', 'Survived'], axis=1)
    y = train['Survived']
    X_final_test = test.drop(['PassengerId', 'Name', 'Surname', 'Ticket', 'Sex', 'Survived'], axis=1).copy()

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)


    # max_score = 0
    # best_forest = None
    # best_params = None
    # for i in range(0, 100):
    #     t1 = time.time()
    #     forest = ensemble.RandomForestClassifier(max_features='sqrt')
    #     parameter_grid = {
    #         'max_depth': [3, 4, 5, 6, 7, 8],
    #         'n_estimators': [100, 150, 200, 250, 300],
    #         'criterion': ['gini', 'entropy'],
    #         'min_samples_leaf' : [2, 3, 4, 5, 6],
    #         'class_weight' : [{0:0.6,1:0.4}, {0:0.5,1:0.5}, {0:0.745,1:0.255}],
    #     }
    #     cross_validation = model_selection.StratifiedKFold(n_splits=5)
    #     grid_search = model_selection.GridSearchCV(forest,param_grid=parameter_grid,cv=cross_validation,n_jobs=-1)
    #     grid_search.fit(x_train, y_train)
    #
    #     score = grid_search.best_estimator_.score(x_test, y_test)
    #     print("耗时 :", time.time() - t1)
    #     print("score : ", score)
    #     print("params : ", grid_search.best_params_)
    #     if max_score < score:
    #         max_score = score
    #         best_forest = grid_search.best_estimator_
    #         best_params = grid_search.best_params_
    # print("max_score : ", max_score)
    # print("best_params : ", best_params)
    # features = pd.DataFrame()
    # features['Feature'] = x_train.columns
    # features['importance'] = best_forest.feature_importances_
    # print(features[['Feature', 'importance']].groupby(['Feature'], as_index=False).mean().sort_values(by='importance',
    #
    #                                                                                         ascending=False))
    # Y_pred = best_forest.predict(X_final_test)


    forest = ensemble.RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0: 0.745, 1: 0.255})
    forest.fit(x, y)
    Y_pred = forest.predict(X_final_test)

    submission = pd.DataFrame({"PassengerId":test["PassengerId"], "Survived":Y_pred})
    submission.to_csv("../data/submission.csv", index=False)
