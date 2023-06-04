from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import preprocessing, model_selection, metrics
import numpy as np

def gpr(df_train, df_test, features, target):
    # Standardize the data
    standardizer = preprocessing.StandardScaler()
    standardizer.fit(df_train[features])

    # Hyperparameter Gridsearch
    kernel_options = []
    for i in np.logspace(0, 10, 10):
        for j in np.logspace(0, 1, 10):
            kernel_options.append((i, j))

    grid = dict()
    grid['alpha'] = [0, 0.01, 0.1, 0.5, 1]
    grid['kernel'] = [RBF(i) + WhiteKernel(j) for i, j in kernel_options]
    grid['normalize_y'] = [True, False]
    gpr_search = model_selection.GridSearchCV(GaussianProcessRegressor(), grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=4)
    gpr_search.fit(standardizer.transform(df_train[features]), df_train[target])
    kernel = gpr_search.best_params_['kernel']
    alpha = gpr_search.best_params_['alpha']
    normalize_y = gpr_search.best_params_['normalize_y']

    # Define Model and Train
    gpr_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=normalize_y)
    gpr_model.fit(standardizer.transform(df_train[features]), df_train[target])

    # Calculating Metrics (MSE and Percent Error)
    def calc_metrics(df_metric):
        preds = gpr_model.predict(standardizer.transform(df_metric[features]))
        if target == 'log_cycle_life':
            preds = np.exp(preds)
        rmse = metrics.mean_squared_error(y_true=df_metric['cycle_life'], y_pred=preds, squared=False)
        precent_error = metrics.mean_absolute_error(y_true=df_metric['cycle_life'], y_pred=preds)*100
        return (rmse, precent_error, preds)

    train_metrics = calc_metrics(df_train)
    test_metrics = calc_metrics(df_test)

    # # Plot observed vs. predicted cycle life
    # title = f'GPR [Kernel=\n{kernel},\n alpha={alpha}] \n Feature Set {dataset}'
    # plotting(df_train, df_val, df_test, train_metrics, val_metrics, test_metrics, title)

    return train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]