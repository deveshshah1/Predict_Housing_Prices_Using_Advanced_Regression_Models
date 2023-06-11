"""
This file represents the various helper functions that were developed to help standardize the way we run
the different models and train them. We organized them here to keep the code cleaner to read.

The functions included in this document include:
1. calc_metrics
2. plot_pred_vs_actual
3. plot_feature_importance

Models to consider adding in the future: TabNet [2]
Model Feature Importance Methods to consider in the future: [1]

[1] Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting and understanding
deep neural networks." Digital signal processing 73 (2018): 1-15.
[2] Arik, Sercan Ö., and Tomas Pfister. "Tabnet: Attentive interpretable tabular learning."
Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 8. 2021.
"""
from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost as xgb
import torch


# Global Parameters for All Models
cv_folds = 4
cv_njobs = -1
cv_scoring = 'neg_root_mean_squared_error'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calc_metrics(model, df, features, target, type='SkLearn', model_input=None):
    """
    This method calculates both the RMSE and %Error of the data in df using the trained model.

    :param model: A trained model that has a valid predict function
    :param df: A dataframe of the data (train or test) that will be used to calculate the stats
    :param features: List of strings of the features we want to use to train the model
    :param target: String of name of target feature we hope to predict in our model
    :param type: String variable that indicates if the model is XGBoost ("Xgb"), ANN ("ANN"), or an Sklearn model
           ("Sklearn"). This determines the format for running model predictions
    :param model_input: Only used for ANN predictions and includes torch tensor of input features for model predictions.
           If not being used, assigned the value None.
    :return: A tuple of the RMSE, %Error, and a list of the predicted target values
    """
    if type == "Xgb":
        dmatrix = xgb.DMatrix(df[features], df[target])
        predictions = model.predict(dmatrix)
    elif type == 'ANN':
        predictions = model(model_input)
        predictions = predictions.cpu().detach().numpy()
        predictions = predictions.squeeze()
    else:
        predictions = model.predict(df[features])
    rmse = metrics.mean_squared_error(y_true=df[target], y_pred=predictions, squared=False)
    percent_error = metrics.mean_absolute_percentage_error(y_true=df[target], y_pred=predictions) * 100
    return rmse, percent_error, predictions


def plot_pred_vs_actual(df_train, df_test, train_metrics, test_metrics, target, title):
    """
    Plots the predicted values of the target (using the trained model) vs. the actual values for both the training
    and testing data (shown in different colors). Also incudes text on the plot showing the RMSE and %Error.
    Additionally, we include a histogram in the corner that shows the residuals of all the predictions-actual values.

    :param df_train: Pandas dataframe of the training data. Includes both features and target variables.
    :param df_test: Pandas dataframe of the testing data. Includes both features and target variables
    :param train_metrics: Tuple of (RMSE, %Error, List of Predictions) for the training dataset. This is the output
           from the calc_metrics function when run on the df_train.
    :param test_metrics: Tuple of (RMSE, %Error, List of Predictions) for the testing dataset. This is the output
           from the calc_metrics function when run on the df_test.
    :param target: String of name of target feature we hope to predict in our model
    :param title: String representing title of graph
    :return: None
    """
    # Add data points
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df_train[target], train_metrics[2], color='blue', label='train')
    ax.scatter(df_test[target], test_metrics[2], color='red', label='test')

    # Add diagnol line for perfect model
    ax.axline((0, 0), slope=1)

    # Add histogram of residuals
    ax2 = fig.add_axes([0.65, 0.2, 0.2, 0.2])
    hist_plot = (list(df_train[target] - train_metrics[2]) + list(df_test[target] - test_metrics[2]))
    hist_plot_abs = [abs(x) for x in hist_plot]
    hist_bounds = max(hist_plot_abs) * 1.2
    ax2.hist(hist_plot, bins=20)
    ax2.set_xlim(left=-hist_bounds, right=hist_bounds)

    # Set labels
    bounds = max(max(df_train[target]), max(train_metrics[2]), max(df_test[target]), max(test_metrics[2])) * 1.2
    ax.set(xlabel=f'Observed {target}', ylabel=f'Predicted {target}', title=title)
    ax.set_ylim(bottom=0, top=bounds)
    ax.set_xlim(left=0, right=bounds)

    # Add text of error values
    ax.text(int(bounds*0.1), int(bounds*0.8), f'Train RMSE: {round(train_metrics[0])}   %Err: {round(train_metrics[1], 1)}\nTest RMSE: {round(test_metrics[0])}   %Err: {round(test_metrics[1], 1)}', color='black')

    # Add legend and show plot
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(features, weights):
    """
    Plots feature relative importances based on the provided weights from the model. Currently this is only being used
    for the linear model as it provides weights for each feature.

    :param features: List of strings of the features we want to use to train the model
    :param weights: List of floats of the model weights of each feature representing importance in the model
    :return: None
    """
    # Make weights all on a relative scale of percentage importance
    weights_standardized = weights / abs(weights).sum()

    # Label (+) correlations as blue and (-) correlations as red
    colors = {'(+) Correlation': 'blue', '(-) Correlation': 'red'}
    labels = list(colors.keys())
    bar_colors = []
    for w in weights:
        if w > 0: bar_colors.append('blue')
        else: bar_colors.append('red')

    # Make plot of data and show plot
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.bar(features, abs(weights_standardized), color=bar_colors)
    ax.set(xlabel='Variable', ylabel='Relative Model Weights', title='Linear Reg. Feature Importance')
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    ax.legend(handles, labels, loc='upper right')
    plt.draw()
    ax.set_xticklabels(features, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
