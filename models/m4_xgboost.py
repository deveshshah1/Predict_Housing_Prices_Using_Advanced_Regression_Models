"""
XGBoost Model
"""
import xgboost as xgb
import numpy as np
from models.model_helper_fxns import *


def run_xgboost(df_train, df_test, features, target, feature_grouping='All Features'):
    """
    Implementation of a XGBoost model. Performs extensive hyper-parameter search to find best number of
    boosting rounds using early stopping.
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
    dtrain_reg = xgb.DMatrix(df_train[features], df_train[target])
    params = {'objective': 'reg:squarederror', 'tree_method': 'hist'}
    results = xgb.cv(params=params, dtrain=dtrain_reg, num_boost_round=5000, nfold=cv_folds, early_stopping_rounds=5)

    # Train Model
    n_estimators = len(results)
    xgb_model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=n_estimators)

    # Calculating Metrics (MSE and Percent Error)
    train_metrics = calc_metrics(xgb_model, df_train, features, target, use_xgb=True)
    test_metrics = calc_metrics(xgb_model, df_test, features, target, use_xgb=True)

    # Plot observed vs. predicted cycle life
    title = f'XGBoost [n_estimators={n_estimators}] \n Feature Set: {feature_grouping}'
    plot_pred_vs_actual(df_train, df_test, train_metrics, test_metrics, target, title)

    return [train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]]
