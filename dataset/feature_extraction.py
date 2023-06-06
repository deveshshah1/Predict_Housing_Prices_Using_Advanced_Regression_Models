import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, r_regression, mutual_info_regression, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def corr_matrix(df_train, features, print_max_n_corr=5):
    df_train = df_train[features]
    orig_corr_mat = df_train.corr(method='pearson')
    orig_corr_mat = np.array(orig_corr_mat)
    corr_mat = np.array(abs(orig_corr_mat))

    for i in range(len(features)):
        corr_mat[i][i] = 0

    max_idx_pairs = []
    for i in range(print_max_n_corr):
        max_loc = np.unravel_index(corr_mat.argmax(), corr_mat.shape)
        max_idx_pairs.append(max_loc)
        corr_mat[max_loc[0]][max_loc[1]] = 0
        corr_mat[max_loc[1]][max_loc[0]] = 0

    toprint = f'---------------------- Top {print_max_n_corr} Pearson R Correlations ----------------------\n'
    for i, j in max_idx_pairs:
        toprint += f'{features[i]} and {features[j]} -> Corr: {round(orig_corr_mat[i][j], 2)}\n'
    print(toprint)


def pca(df_train, features, n_components=1, obtain_top_perct=95):
    df_train_pca = df_train[features]
    pca = PCA(n_components=n_components)
    pca.fit(df_train_pca)
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

    toprint = f'---------------- PCA Components for {obtain_top_perct}% Explained Variance -----------------\n'
    for i in range(top_k+1):
        toprint += f'PCA #{i} Explained Variance: {round(explained_variance[i], 2)}, Cumulative Variance: {round(cumulative_variance[i], 2)}\n'
    print(toprint)

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

    # Return top K principle components for each
    df_pca = pd.DataFrame(pca.transform(df_train_pca), columns=[f'PCA_{i}' for i in range(n_components)])
    df_pca = df_pca.drop(columns=[f'PCA_{i}' for i in range(top_k+1, n_components)])
    df_train_with_pca = pd.concat([df_train, df_pca], axis=1)
    return df_train_with_pca


def univariate_feature_selection(df_train, features, target, k=1):
    y = df_train[target]
    x = df_train[features]
    results = {}

    select = SelectKBest(score_func=f_regression, k=k)
    z = select.fit(x, y)
    f_reg_scores = z.scores_
    f_reg_scores_normalized = (f_reg_scores - min(f_reg_scores)) / (max(f_reg_scores) - min(f_reg_scores))
    f_reg_scores_sorted_idx = np.argsort(np.array(f_reg_scores_normalized))
    scoring = [0]*k
    for i, j in enumerate(f_reg_scores_sorted_idx):
        scoring[j] = k-i
    results['Univariate Feat. Selection w/ F Regression'] = scoring

    select = SelectKBest(score_func=r_regression, k=k)
    z = select.fit(x, y)
    r_reg_scores = z.scores_
    r_reg_scores_normalized = (r_reg_scores - min(r_reg_scores)) / (max(r_reg_scores) - min(r_reg_scores))
    r_reg_scores_sorted_idx = np.argsort(np.array(r_reg_scores_normalized))
    scoring = [0]*k
    for i, j in enumerate(r_reg_scores_sorted_idx):
        scoring[j] = k-i
    results['Univariate Feat. Selection w/ R Regression'] = scoring

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
    y = df_train[target]
    x = df_train[features]
    results = {}

    model = SVR(kernel='linear') #to use diff kernel need https://stats.stackexchange.com/questions/191402/feature-selection-using-rfe-in-svm-kernel-other-than-linear-eg-rbf-poly-etc
    rfe = RFE(model, n_features_to_select=1)
    fit = rfe.fit(x, y)
    svr_support = fit.support_
    svr_ranking = fit.ranking_
    results['Recursive Feat. Elimination w/SVR Linear Kernel'] = svr_ranking
    return results


def tree_based_feature_selection(df_train, features, target):
    y = df_train[target]
    x = df_train[features]
    results = {}

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
