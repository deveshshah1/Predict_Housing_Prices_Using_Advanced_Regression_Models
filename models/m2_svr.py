"""
Multi-variate Support Vector Regression (SVR) Model
"""
from sklearn import model_selection, svm
import numpy as np
from models.model_helper_fxns import *


def run_svr(df_train, df_test, features, target, feature_grouping='All Features'):
    """
    Implementation of a SVR model. Performs extensive hyper-parameter grid search
    for best model using n-fold cross-validation on training set.
    Metrics are reported as RMSE and %Error in both table and graph.

    :param df_train: Pandas dataframe of the training data. Includes both features and target variables.
    :param df_test: Pandas dataframe of the testing data. Includes both features and target variables.
    :param features: List of strings of the features we want to use to train the model
    :param target: String of name of target feature we hope to predict in our model
    :param feature_grouping: String representing name of feature set. Default = 'All Features". Meant
           to serve as identifier on graphed results as we vary input feature sets
    :return: List in the following format: [train rmse, train percent error, test rmse, test percent error]
    """
    # Hyper-parameter Search
    grid = dict()
    grid['gamma'] = ['scale', 'auto']
    grid['kernel'] = ['rbf', 'linear', 'poly']
    grid['C'] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    grid['epsilon'] = np.arange(0, 1, 0.01)
    svr_search = model_selection.GridSearchCV(svm.SVR(), grid, scoring=cv_scoring, n_jobs=cv_njobs, cv=cv_folds)
    svr_search.fit(df_train[features], df_train[target])
    gamma = svr_search.best_params_['gamma']
    kernel = svr_search.best_params_['kernel']
    C = svr_search.best_params_['C']
    epsilon = svr_search.best_params_['epsilon']

    # Define Model and Train
    svr_model = svm.SVR(kernel=kernel, gamma=gamma, C=C, epsilon=epsilon)
    svr_model.fit(df_train[features], df_train[target])

    # Calculating Metrics (MSE and Percent Error)
    train_metrics = calc_metrics(svr_model, df_train, features, target)
    test_metrics = calc_metrics(svr_model, df_test, features, target)

    # Plot observed vs. predicted cycle life
    title = f'SVR [kernel={kernel}, gamma={gamma}, C={C}, epsilon={epsilon}] \n Feature Set: {feature_grouping}'
    plot_pred_vs_actual(df_train, df_test, train_metrics, test_metrics, target, title)

    return [train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]]
