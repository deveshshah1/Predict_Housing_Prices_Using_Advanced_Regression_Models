import xgboost as xgb
import numpy as np
from models.model_helper_fxns import *


def xgboost(df_train, df_test, features, target, feature_grouping='All Features'):
    # Hyper-parameter Search
    dtrain_reg = xgb.DMatrix(df_train[features], df_train[target])
    params = {'objective': 'reg:squarederror', 'tree_method': 'hist'}
    results = xgb.cv(params=params, dtrain=dtrain_reg, num_boost_round=5000, nfold=cv_folds, early_stopping_rounds=5)

    # Train Model
    n_estimators = len(results)
    xgb_model = xgb.train(params=params, dtrain=dtrain_reg, num_boost_round=n_estimators)

    # Calculating Metrics (MSE and Percent Error)
    train_metrics = calc_metrics(xgb_model, df_train, features, target, xgb=True)
    test_metrics = calc_metrics(xgb_model, df_test, features, target, xgb=True)

    # Plot observed vs. predicted cycle life
    title = f'XGBoost [n_estimators={n_estimators}] \n Feature Set: {feature_grouping}'
    plot_pred_vs_actual(df_train, df_test, train_metrics, test_metrics, target, title)

    return [train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]]
