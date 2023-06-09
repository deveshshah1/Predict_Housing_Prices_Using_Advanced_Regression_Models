"""
Random Forest Model
"""
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from models.model_helper_fxns import *


def run_random_forest(df_train, df_test, features, target, feature_grouping='All Features'):
    """
    Implementation of a Random Forest model. Performs extensive hyper-parameter grid search
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
    grid['n_estimators'] = [20, 50, 100, 200, 400]
    grid['max_depth'] = [5, 10, 20, None]
    grid['min_samples_split'] = [2, 4]
    grid['min_samples_leaf'] = [1, 2]
    grid['max_features'] = ['sqrt']
    randForest_search = model_selection.GridSearchCV(RandomForestRegressor(), grid, scoring=cv_scoring, n_jobs=cv_njobs, cv=cv_folds)
    randForest_search.fit(df_train[features], df_train[target])
    n_estimators = randForest_search.best_params_['n_estimators']
    max_depth = randForest_search.best_params_['max_depth']
    min_samples_split = randForest_search.best_params_['min_samples_split']
    min_samples_leaf = randForest_search.best_params_['min_samples_leaf']
    max_features = randForest_search.best_params_['max_features']

    # Define Model and Train
    randForest_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
    randForest_model.fit(df_train[features], df_train[target])

    # Calculating Metrics (MSE and Percent Error)
    train_metrics = calc_metrics(randForest_model, df_train, features, target)
    test_metrics = calc_metrics(randForest_model, df_test, features, target)

    # Plot observed vs. predicted cycle life
    title = f'Random Forest [n_estimators={n_estimators}, max_depth={max_depth}, \nmin_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, \nmax_features={max_features}] \n Feature Set: {feature_grouping}'
    plot_pred_vs_actual(df_train, df_test, train_metrics, test_metrics, target, title)

    return [train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]]
