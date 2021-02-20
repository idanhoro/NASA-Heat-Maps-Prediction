import numpy as np
import pandas as pd
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
from time import time


def color_predict(df):
    x = df[df.columns[(df.columns == 'X') | (df.columns == 'Y') | (df.columns == 'Year') | (df.columns == 'Month') | (
            df.columns == 'Snow Cover Is Valuable')]]
    y = df['Snow Cover Color Index']
    x_test = x.loc[(x['Year'] == 2020) & (x['Month'] == 12)]
    x_train = x.loc[(x['Year'] != 2020) | (x['Month'] != 12)]
    y_test = y.loc[(x['Year'] == 2020) & (x['Month'] == 12)]
    y_train = y.loc[(x['Year'] != 2020) | (x['Month'] != 12)]

    # x_train['Year'] = x_train['Year'].sub(2000)
    # x_test['Year'] = x_test['Year'].sub(2000)
    Compare_models(x_train, x_test, y_train, y_test)


def Compare_models(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x_scale_train = scaler.fit_transform(x_train)
    x_scale_test = scaler.transform(x_test)

    Linear_Regression_clf = Linear_Regression(x_scale_train, y_train)
    Linear_Pipeline_clf = Linear_Pipeline(x_scale_train, y_train)
    Linear_BayesianRidge_clf = Linear_BayesianRidge(x_scale_train, y_train)
    Linear_Perceptron_clf = Linear_Perceptron(x_scale_train, y_train)
    Naive_Bayes_clf = Naive_Bayes(x_scale_train, y_train)

    Linear_Score(Linear_Regression_clf, x_scale_train, x_scale_test, y_test, y_train, "Linear_Regression")
    Linear_Score(Linear_Pipeline_clf, x_scale_train, x_scale_test, y_test, y_train, "Linear_Pipeline")
    Linear_Score(Linear_BayesianRidge_clf, x_scale_train, x_scale_test, y_test, y_train, "Linear_BayesianRidge")
    Linear_Score(Linear_Perceptron_clf, x_scale_train, x_scale_test, y_test, y_train, "Linear_Perceptron")
    Linear_Score(Naive_Bayes_clf, x_scale_train, x_scale_test, y_test, y_train, "Naive_Bayes")

    # parameter = {
    #     'hidden_layer_sizes': [(10, 30, 10), (20,)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 1e-5],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    # parameter = {'activation': ['relu'], 'alpha': [0.0001], 'hidden_layer_sizes': [(20,)],
    #              'learning_rate': ['constant'], 'solver': ['adam']}
    # neural_network_clf, scaler = neural_network(x_train, y_train)
    # score_compare(neural_network_clf, 'Neural Network', scaler.transform(x_train), scaler.transform(x_test), y_train,
    #               y_test)


def Linear_Score(clf, x_train, x_test, y_test, y_train, model_name):
    print("#################################")
    start_time = time()
    print("Start at: {}".format(start_time))
    print(model_name)
    train_score = clf.score(x_train, y_train)
    print("Train Score:{}".format(train_score))
    test_score = clf.score(x_test, y_test)
    print("Test Score:{}".format(test_score))
    print("Finished at: {}".format(time()))
    print("--- {} seconds ---".format(time() - start_time))
    return train_score, test_score


def Naive_Bayes(x_train, y_train):
    gnb = GaussianNB().fit(x_train, y_train)
    return gnb


def neural_network(x_train, y_train, parameters=None):
    clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
    scaler = StandardScaler().fit(x_train)
    years = x_train['Year'].unique()
    months = x_train['Month'].unique()
    for year in years:
        for month in months:
            x = x_train.loc[(x_train['Year'] == year) & (x_train['Month'] == month)]
            y = y_train.loc[(x_train['Year'] == year) & (x_train['Month'] == month)]
            x_scale = scaler.transform(x)
            clf.partial_fit(x_scale, y)
    return clf, scaler


def Linear_Regression(x_train, y_train):
    reg = LinearRegression().fit(x_train, y_train)
    return reg


def Linear_BayesianRidge(x_train, y_train):
    reg = linear_model.BayesianRidge().fit(x_train, y_train)
    return reg


def Linear_Pipeline(x_train, y_train):
    model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))]).fit(
        x_train, y_train)
    return model


def Linear_Perceptron(x_train, y_train):
    clf = Perceptron(fit_intercept=False, max_iter=10, tol=None, shuffle=False).fit(x_train, y_train)
    return clf

# def DL_model():
#     df = pd.read_csv(r"./CSVs/2021-02-18T14_11_18.csv")
#     X = df[df.columns[(df.columns == 'X') | (df.columns == 'Y') | (df.columns == 'Year') | (df.columns == 'Month') | (
#             df.columns == 'Vegetation Is Valuable')]]
#     y = df['Vegetation Color Index']
#     X_test = X.loc[(X['Year'] == 2020) & (X['Month'] == 12)]
#     X_train = X.loc[(X['Year'] != 2020) | (X['Month'] != 12)]
#     y_test = y.loc[(X['Year'] == 2020) & (X['Month'] == 12)]
#     y_train = y.loc[(X['Year'] != 2020) | (X['Month'] != 12)]
#     scaler = StandardScaler()
#     scaler.fit(X_train)  # Don't cheat - fit only on training data
#     X_train = scaler.transform(X_train)
#     X_test = scaler.transform(X_test)
#     clf = SGDClassifier(alpha=.0001, loss='log', penalty='l2', n_jobs=-1, shuffle=True, max_iter=100, verbose=0,
#                         tol=0.001)
#
#     ROUNDS = 6
#     for _ in range(ROUNDS):
#         batcherator = batch(X_train, y_train, 10)
#         for index, (chunk_X, chunk_y) in enumerate(batcherator):
#             clf.partial_fit(chunk_X, chunk_y)
#
#             y_predicted = clf.predict(X_test)
#             print(accuracy_score(y_test, y_predicted))
#
#
# def batch(iterable_X, iterable_y, n=1):
#     l = len(iterable_X)
#     for ndx in range(0, l, n):
#         yield iterable_X[ndx:min(ndx + n, l)], iterable_y[ndx:min(ndx + n, l)]
