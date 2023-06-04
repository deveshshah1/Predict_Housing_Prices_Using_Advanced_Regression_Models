from sklearn import model_selection, svm
import numpy as np
from models.model_helper_fxns import *


def run_svr(df_train, df_test, features, target, feature_grouping='All Features'):
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
