"""
Gaussian Process Regression (GPR) Model
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import model_selection
import numpy as np
from models.model_helper_fxns import *


def run_gpr(df_train, df_test, features, target, feature_grouping='All Features'):
    """
    Implementation of a GPR model. Performs extensive hyper-parameter grid search
    for best model using n-fold cross-validation on training set.
    The GPR model has more potential using more advanced techniques to modify the kernels to the specific
    dataset. Consider using GPy or BoTorch in the future depending on the use case.
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
