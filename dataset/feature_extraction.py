"""
This file performs a wide array of feature extraction studies. We print out the results from various methods
and let the user decide which features/how many to select moving forward.In particular we address methods such as
PCA, Pearson-R Correlation, Recursive Feature Elimination, Univariate Feature Selection and
Tree Based Feature Selection.

These implementations are all derived from the sklearn site. Refer here for more options:
https://scikit-learn.org/stable/modules/feature_selection.html


Model Feature Importance Methods to consider in the future: [1]

[1] Montavon, Grégoire, Wojciech Samek, and Klaus-Robert Müller. "Methods for interpreting and understanding
deep neural networks." Digital signal processing 73 (2018): 1-15.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, r_regression, mutual_info_regression, RFE
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def corr_matrix(df_train, features, print_max_n_corr=5):
    """
    Builds a correlation matrix for every pair of input features using the pearson r correlation.
    Identifies the n (user defined) highest pairs of correlation and prints out these feature pairs.

    :param df_train: Pandas dataframe of the training data. Includes both features and target variables.
    :param features: List of strings of the features we want to use to train the model
    :param print_max_n_corr: Int representing the num of highest correlation pair to print out
    :return: None
    """
    # Build correlation matrix for training data
    df_train = df_train[features]
    orig_corr_mat = df_train.corr(method='pearson')
    orig_corr_mat = np.array(orig_corr_mat)
    corr_mat = np.array(abs(orig_corr_mat))

    # Identify max correlation pairs (not counting self-pairs)
    for i in range(len(features)):
        corr_mat[i][i] = 0

    max_idx_pairs = []
    for i in range(print_max_n_corr):
        max_loc = np.unravel_index(corr_mat.argmax(), corr_mat.shape)
        max_idx_pairs.append(max_loc)
        corr_mat[max_loc[0]][max_loc[1]] = 0
        corr_mat[max_loc[1]][max_loc[0]] = 0

    # Print our results
    toprint = f'---------------------- Top {print_max_n_corr} Pearson R Correlations ----------------------\n'
    for i, j in max_idx_pairs:
        toprint += f'{features[i]} and {features[j]} -> Corr: {round(orig_corr_mat[i][j], 2)}\n'
    print(toprint)


def pca(df_train, features, n_components=1, obtain_top_perct=95):
    """
    Here.

    :param df_train: Pandas dataframe of the training data. Includes both features and target variables.
    :param features: List of strings of the features we want to use to train the model
    :param n_components: Int representing the number of principle components to find
    :param obtain_top_perct: Int (0-100) representing what percent of the variation in the data we hope to keep. This
                             then becomes the number of principle components to keep.
    :return: df_train_with_pca: pandas dataframe representing the top k principle components (determined by
             obtain_top_perct) appended to the original df_train
    """
    # Perform PCA on our training data
    df_train_pca = df_train[features]
    pca = PCA(n_components=n_components)
    pca.fit(df_train_pca)

    # Calculate the cumulative variance to find k = the number of PCA components to reach obtain_top_perct
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = [0] * n_components
    pca_nums = list(range(0, n_components))
    top_k = 0
    for i, var in enumerate(explained_variance):
        if i > 0:
            cumulative_variance[i] = cumulative_variance[i-1] + var
            if cumulative_variance[i] > 0.95 and cumulative_variance[i - 1] < (obtain_top_perct/100): top_k = i
        else:
            cumulative_variance[i] = var

    # Print our top k components and their associated variances
    toprint = f'---------------- PCA Components for {obtain_top_perct}% Explained Variance -----------------\n'
    for i in range(top_k+1):
        toprint += f'PCA #{i} Explained Variance: {round(explained_variance[i], 2)}, Cumulative Variance: {round(cumulative_variance[i], 2)}\n'
    print(toprint)

    # Plot all principle components along with labels for top k
    fig, ax1 = plt.subplots(figsize=(5,5))
    ax2 = plt.twinx(ax=ax1)
    ax1.bar(pca_nums, explained_variance)
    ax2.plot(pca_nums, cumulative_variance, color='red')
    for i in range(top_k+1):
        ax1.text(i, explained_variance[i], round(cumulative_variance[i], 2), ha='center', color='red')
    ax1.set(xlabel='Principal Component #', ylabel='Explained Variance', title=f'Selecting PCA K={top_k+1} Components for {obtain_top_perct}% Variance')
    ax2.set(ylabel='Cumulative Variance')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')
    plt.tight_layout()
    plt.show()

    # Return top K principle components appended onto our original df_train
    df_pca = pd.DataFrame(pca.transform(df_train_pca), columns=[f'PCA_{i}' for i in range(n_components)])
    df_pca = df_pca.drop(columns=[f'PCA_{i}' for i in range(top_k+1, n_components)])
    df_train_with_pca = pd.concat([df_train, df_pca], axis=1)
    return df_train_with_pca


def univariate_feature_selection(df_train, features, target, k=1):
    """
    Performs univariate feature selection using three scoring functions: f_regression, r_regression, and mutual
    information. Obtains a scoring for each feature of how important it is and returns the ordered list of most
    important features (1 = most important)

    :param df_train: Pandas dataframe of the training data. Includes both features and target variables.
    :param features: List of strings of the features we want to use to train the model
    :param target: String of name of target feature we hope to predict in our model
    :param k: Int representing k in selectKbest. Set to len(features) by default to score all the features
    :return: results: Dictionary of results stored by method type
    """
    # Obtain x and y inputs from df_train
    y = df_train[target]
    x = df_train[features]
    results = {}

    # Run SelectKBest on f_regression. Put ordered results of feature importance as list into dictionary
    select = SelectKBest(score_func=f_regression, k=k)
    z = select.fit(x, y)
    f_reg_scores = z.scores_
    f_reg_scores_normalized = (f_reg_scores - min(f_reg_scores)) / (max(f_reg_scores) - min(f_reg_scores))
    f_reg_scores_sorted_idx = np.argsort(np.array(f_reg_scores_normalized))
    scoring = [0]*k
    for i, j in enumerate(f_reg_scores_sorted_idx):
        scoring[j] = k-i
    results['Univariate Feat. Selection w/ F Regression'] = scoring

    # Run SelectKBest on r_regression. Put ordered results of feature importance as list into dictionary
    select = SelectKBest(score_func=r_regression, k=k)
    z = select.fit(x, y)
    r_reg_scores = z.scores_
    r_reg_scores_normalized = (r_reg_scores - min(r_reg_scores)) / (max(r_reg_scores) - min(r_reg_scores))
    r_reg_scores_sorted_idx = np.argsort(np.array(r_reg_scores_normalized))
    scoring = [0]*k
    for i, j in enumerate(r_reg_scores_sorted_idx):
        scoring[j] = k-i
    results['Univariate Feat. Selection w/ R Regression'] = scoring

    # Run SelectKBest on mutual info. Put ordered results of feature importance as list into dictionary
    select = SelectKBest(score_func=mutual_info_regression, k=k)
    z = select.fit(x, y)
    mutual_reg_scores = z.scores_
    mutual_reg_scores_normalized = (mutual_reg_scores - min(mutual_reg_scores)) / (max(mutual_reg_scores) - min(mutual_reg_scores))
    mutual_reg_scores_sorted_idx = np.argsort(np.array(mutual_reg_scores_normalized))
    scoring = [0]*k
    for i, j in enumerate(mutual_reg_scores_sorted_idx):
        scoring[j] = k-i
    results['Univariate Feat. Selection w/ Mutual Info'] = scoring

    return results


def recursive_feature_elim(df_train, features, target):
    """
    Performs recursive feature selection using SVR and a linear kernel. Obtains a scoring for each feature of
    how important it is and returns the ordered list of most important features (1 = most important).
    Note: Other model types or kernels (like 'rbf') were not used since they don't have a built in coef_ or
    importance_ variable that provides importance scores. This would need to be manually configured. See:
    https://link.springer.com/article/10.1007/s10462-011-9205-2

    :param df_train: Pandas dataframe of the training data. Includes both features and target variables.
    :param features: List of strings of the features we want to use to train the model
    :param target: String of name of target feature we hope to predict in our model
    :return: results: Dictionary of results stored by method type
    """
    # Obtain x and y inputs from df_train
    y = df_train[target]
    x = df_train[features]
    results = {}

    # Implement RFE on SVR model. Put ordered results of feature importance as list into dictionary
    model = SVR(kernel='linear')
    rfe = RFE(model, n_features_to_select=1)
    fit = rfe.fit(x, y)
    svr_support = fit.support_
    svr_ranking = fit.ranking_
    results['Recursive Feat. Elimination w/SVR Linear Kernel'] = svr_ranking
    return results


def tree_based_feature_selection(df_train, features, target):
    """
    Performs tree based feature selection using extra trees regressor. Obtains a scoring for each feature of
    how important it is and returns the ordered list of most important features (1 = most important).

    :param df_train: Pandas dataframe of the training data. Includes both features and target variables.
    :param features: List of strings of the features we want to use to train the model
    :param target: String of name of target feature we hope to predict in our model
    :return: results: Dictionary of results stored by method type
    """
    # Obtain x and y inputs from df_train
    y = df_train[target]
    x = df_train[features]
    results = {}

    # Implement Tree Based Feat Selection. Put ordered results of feature importance as list into dictionary
    model = ExtraTreesRegressor(n_estimators=50)
    model.fit(x, y)
    feat = model.feature_importances_
    feat_sorted_idx = np.argsort(np.array(feat))
    k = len(features)
    scoring = [0]*k
    for i, j in enumerate(feat_sorted_idx):
        scoring[j] = k-i
    results['Tree Based Feat. Selection w/ExtraTreesRegressor'] = scoring
    return results


def perform_feature_extraction(df_train, features, target):
    """
    This method runs all the feature extraction techniques highlighted in this document.
    It prints out the results for each technique in a clean manner to the user.
    Note nothing is returned. But if desired, the df_train_with_pca can be returned and used if a training with
    principle components is desired.

    :param df_train: Pandas dataframe of the training data. Includes both features and target variables.
    :param features: List of strings of the features we want to use to train the model
    :param target: String of name of target feature we hope to predict in our model
    :return: None
    """
    corr_matrix(df_train, features)
    df_train_with_pca = pca(df_train, features, n_components=len(features))
    results = univariate_feature_selection(df_train, features, target, k=len(features))
    rfe_results = recursive_feature_elim(df_train, features, target)
    results.update(rfe_results)
    trees_results = tree_based_feature_selection(df_train, features, target)
    results.update(trees_results)

    df_results = pd.DataFrame(results)
    df_results.index = features
    df_results = df_results.T
    print(f'----------- Feature Selection Methods (lower scorer = more important feature) ------------\n')
    print(df_results)
    print(f'------------------------------------------------------------------------------------------\n')
