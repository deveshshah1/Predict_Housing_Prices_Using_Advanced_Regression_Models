from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from models.model_helper_fxns import *


def random_forest(df_train, df_test, features, target, feature_grouping='All Features'):
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
