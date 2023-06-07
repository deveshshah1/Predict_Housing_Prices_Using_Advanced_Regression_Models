"""
This file represents the loading, preprocessing, and statistical calculations for the data.
There are three functions:
1. preprocess_data: Loads the data, normalizes it, and splits into train/test
2. dataset_statistics: Prints standard stats about our features and the pearson r correlation matrix
3. visualize_dataset_relationships: Plots the impact of each feature on our target variable

For our dataset in this project, the data is defined as follows:
    Each record in the database describes a Boston suburb or town. The data was drawn from the Boston Standard
    Metropolitan Statistical Area (SMSA) in 1970. The attributes are defined as follows:

    Inputs Features:
    CRIM: per capita crime rate by town
    ZN: proportion of residential land zoned for lots over 25,000 sq. ft.
    INDUS: proportion of non-retail buisness acres per town
    CHAS: Charles River dummy Variable (= 1 if tract bounds river; 0 otherwise)
    NOX: nitric oxides concentration (parts per 10 million)
    RM: averge number of rooms per dwelling
    AGE: proportion of owner-occupied units built prior to 1940
    DIS: weighted distances to five Boston employment centers
    RAD: index of accessibility to radial highways
    TAX: full-value property tax rate per $10K
    PTRATIO: pupil-teacher ratio by town
    B: 1000(Bk-0.63)2 where Bk is the proportion of blacks by town
    LSTAT: pct lower status of the population

    Output Feature:
    MEDV: medivan value of owner-occupied homes in $1000s
"""
import pandas as pd
from sklearn import model_selection, preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import math


def preprocess_data(dataset_path, features, target, process='abs_scaling', random_state=42):
    """
    Loads the dataset as a pandas dataframe. Splits into train/val/test sets. And normalizes/standardizes each
    feature set based on the desired method.

    :param dataset_path: Absolute path to where dataset excel file is found
    :param features: List of features we want to use to train the model
    :param target: String of name of target feature we hope to predict in our model
    :param process: Type of standardization/preprocessing to perform.
                    Valid options include {abs_scaling, normalization, standardization}
    :param random_state: Int representing the random state in the train/test split
    :return: df_train: pandas dataframe of our features and targets for the training data
             df_test: pandas dataframe of our features and targets for the test data
    """
    # Load Dataset
    df = pd.read_csv(dataset_path)

    # Split into train/test
    df_train, df_test = model_selection.train_test_split(df, test_size=0.2, random_state=random_state)

    # Process the data
    if process == 'abs_scaling':
        # 0-1 Absolute Scaling
        abs_scaler = preprocessing.MaxAbsScaler()
        abs_scaler.fit(df_train[features])
        df_train[features] = abs_scaler.transform(df_train[features])
        df_test[features] = abs_scaler.transform(df_test[features])
    elif process == 'normalization':
        # Normalization
        normalizer = preprocessing.MinMaxScaler()
        normalizer.fit(df_train[features])
        df_train[features] = normalizer.transform(df_train[features])
        df_test[features] = normalizer.transform(df_test[features])
    elif process == 'standardization':
        # Standardization
        standardizer = preprocessing.StandardScaler()
        standardizer.fit(df_train[features])
        df_train[features] = standardizer.transform(df_train[features])
        df_test[features] = standardizer.transform(df_test[features])
    else:
        raise Exception('Invalid Preprocessing Type')

    return df_train, df_test


def dataset_statistics(dataset_path, target):
    """
    Calculates and prints basic statistics for our entire dataset. Also prints out the correlation matrix for all
    variables in the dataset

    :param dataset_path: Absolute path to where dataset excel file is found
    :param target: String of name of target feature we hope to predict in our model
    :return: None
    """
    # Load Dataset
    df = pd.read_csv(dataset_path)

    # Perform Basic Statistics on Dataset
    dtype = list(df.dtypes)
    num_nulls = list(df.isna().sum())
    pearson_correlation = list(df.corr()[target])
    df_statistics = pd.DataFrame([dtype, num_nulls, pearson_correlation])
    df_statistics.columns = df.columns
    df_statistics.index = ['dtype', 'Number_Nulls', 'Pearson_R_Correlation']
    df_statistics = pd.concat([df.describe(), df_statistics])
    print(df_statistics)

    # Visualize Correlation Matrix
    corr_mat = df.corr(method='pearson')
    sns.heatmap(corr_mat, annot=True, annot_kws={'fontsize':5})
    plt.show()


def visualize_dataset_relationships(dataset_path, features, target, random_state):
    """
    Plots the relationship of every feature variable vs. our target variable. This shows us how big of a linear
    correlation there is between each feature to our target we hope to predict.

    :param dataset_path: Absolute path to where dataset excel file is found
    :param features: List of features we want to use to train the model
    :param target: String of name of target feature we hope to predict in our model
    :param random_state: Int representing the random state in the train/test split
    :return: None
    """
    # Load Dataset
    df = pd.read_csv(dataset_path)

    # Split into train/test
    df_train, df_test = model_selection.train_test_split(df, test_size=0.2, random_state=random_state)

    # Plot relationship of all variables vs. target variable
    y_loc = int(max(df[target]) * 0.8)
    num_grids = math.ceil(math.sqrt(len(features)))
    fig, ax = plt.subplots(nrows=num_grids, ncols=num_grids, figsize=(10, 10))
    for i in range(len(features)):
        row = i // num_grids
        col = i % num_grids
        ax[row, col].scatter(df_train[features[i]], df_train[target], color='blue', label='train')
        ax[row, col].scatter(df_test[features[i]], df_test[target], color='red', label='test')
        ax[row, col].set(xlabel=features[i], ylabel=target)

        p_val = df[features[i]].corr(df[target])
        l, r = ax[row, col].get_xlim()
        ax[row, col].text(((r-l)*0.2 + l), y_loc, f'p={round(p_val, 2)}', style='italic')
    handles, labels = ax[0, num_grids-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()
