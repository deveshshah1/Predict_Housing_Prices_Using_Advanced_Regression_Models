from sklearn import model_selection, linear_model
import numpy as np
from models.model_helper_fxns import *


def linear_regression(df_train, df_test, features, target, feature_grouping='All Features'):
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

    return [train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]]
