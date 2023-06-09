"""
Multi-variate Linear Regression Model using ElasticNet Architecture
"""
from sklearn import model_selection, linear_model
import numpy as np
from models.model_helper_fxns import *


def run_linear_regression(df_train, df_test, features, target, feature_grouping='All Features'):
    """
    Implementation of a multivariate linear regression model. Performs extensive hyper-parameter grid search
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
    grid['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 0.1, 1.0, 10.0, 100.0]
    grid['l1_ratio'] = np.arange(0, 1, 0.01)
    elastic_net_search = model_selection.GridSearchCV(linear_model.ElasticNet(), grid, scoring=cv_scoring, n_jobs=cv_njobs, cv=cv_folds)
    elastic_net_search.fit(df_train[features], df_train[target])
    alpha = elastic_net_search.best_params_['alpha']
    l1_ratio = elastic_net_search.best_params_['l1_ratio']

    # Define Model and Train
    elastic_net = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elastic_net.fit(df_train[features], df_train[target])

    # Calculating Metrics (MSE and Percent Error)
    train_metrics = calc_metrics(elastic_net, df_train, features, target)
    test_metrics = calc_metrics(elastic_net, df_test, features, target)

    # Plot observed vs. predicted cycle life
    title = f'Linear Regression [alpha={alpha}, l1_ratio={l1_ratio}] \n Feature Set: {feature_grouping}'
    plot_pred_vs_actual(df_train, df_test, train_metrics, test_metrics, target, title)

    # Plot feature importance
    plot_feature_importance(features, elastic_net.coef_)

    return [train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]]
