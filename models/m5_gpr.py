from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import model_selection
import numpy as np
from models.model_helper_fxns import *


def run_gpr(df_train, df_test, features, target, feature_grouping='All Features'):
    # Hyper-parameter Search
    kernel_options = []
    for i in np.logspace(0, 10, 10):
        for j in np.logspace(0, 1, 10):
            kernel_options.append((i, j))

    kernel_options = [(.1, .1)]

    grid = dict()
    grid['alpha'] = [0, 0.01, 0.1, 0.5, 1]
    grid['kernel'] = [RBF(i) + WhiteKernel(j) for i, j in kernel_options]
    grid['normalize_y'] = [True, False]
    gpr_search = model_selection.GridSearchCV(GaussianProcessRegressor(), grid, scoring=cv_scoring, n_jobs=cv_njobs, cv=cv_folds)
    gpr_search.fit(df_train[features], df_train[target])
    kernel = gpr_search.best_params_['kernel']
    alpha = gpr_search.best_params_['alpha']
    normalize_y = gpr_search.best_params_['normalize_y']

    # Define Model and Train
    gpr_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=normalize_y)
    gpr_model.fit(df_train[features], df_train[target])

    # Calculating Metrics (MSE and Percent Error)
    train_metrics = calc_metrics(gpr_model, df_train, features, target)
    test_metrics = calc_metrics(gpr_model, df_test, features, target)

    # Plot observed vs. predicted cycle life
    title = f'GPR [Kernel=\n{kernel},\n alpha={alpha}] \n Feature Set: {feature_grouping}'
    plot_pred_vs_actual(df_train, df_test, train_metrics, test_metrics, target, title)

    return [train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]]
