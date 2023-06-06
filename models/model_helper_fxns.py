from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost as xgb


# Global Parameters for All Models
cv_folds = 4
cv_njobs = -1
cv_scoring = 'neg_root_mean_squared_error'


def calc_metrics(model, df, features, target, use_xgb=False):
    if use_xgb:
        dmatrix = xgb.DMatrix(df[features], df[target])
        predictions = model.predict(dmatrix)
    else:
        predictions = model.predict(df[features])
    rmse = metrics.mean_squared_error(y_true=df[target], y_pred=predictions, squared=False)
    percent_error = metrics.mean_absolute_percentage_error(y_true=df[target], y_pred=predictions) * 100
    return (rmse, percent_error, predictions)


def plot_pred_vs_actual(df_train, df_test, train_metrics, test_metrics, target, title):
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
    weights_standardized = weights / abs(weights).sum()
    colors = {'(+) Correlation': 'blue', '(-) Correlation': 'red'}
    labels = list(colors.keys())
    bar_colors = []
    for w in weights:
        if w > 0: bar_colors.append('blue')
        else: bar_colors.append('red')

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.bar(features, abs(weights_standardized), color=bar_colors)
    ax.set(xlabel='Variable', ylabel='Relative Model Weights', title='Linear Reg. Feature Importance')
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    ax.legend(handles, labels, loc='upper right')
    plt.draw()
    ax.set_xticklabels(features, rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
