"""
Author: Devesh Shah
Project: Predict Housing Prices Using Advanced Regression Models

In this project, we explore various advanced regression models on smaller tabular datasets. The project is
organized such that we hope it can serve as a reference for others building regression models on tabular data.
We explore ideas from feature selection to visualization of the data to best understand the data.
"""
from dataset.dataset_preprocessing import *
from dataset.feature_extraction import perform_feature_extraction
from models.m1_linear_regression import *
from models.m2_svr import *
from models.m3_random_forest import *
from models.m4_xgboost import *
from models.m5_gpr import *


def main():
    """
    This is our main function that drives the program. It first defines the standard variables we will hold constant.
    Then we load the data and preprocess it. This also involves methods such as feature extraction and finding
    general stats about the data. From here, we use this knowledge to hyper-parameter several simple models.
    The results are printed out to the user.
    """
    # Define features and target variables for dataset ------------------------------------------------------------
    features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    target = 'MEDV'
    random_state = 42
    # pd.options.display.max_columns = None

    # Load and process data ---------------------------------------------------------------------------------------
    dataset_path = './dataset/boston_housing_prices.csv'
    dataset_statistics(dataset_path, target)
    # visualize_dataset_relationships(dataset_path, features, target, random_state)
    df_train, df_test = preprocess_data(dataset_path, features, target, process='abs_scaling', random_state=random_state)
    perform_feature_extraction(df_train, features, target)

    # Run Models --------------------------------------------------------------------------------------------------
    results = {}
    results['Linear_Regression'] = run_linear_regression(df_train, df_test, features, target)
    results['SVR'] = run_svr(df_train, df_test, features, target)
    results['Random_Forest'] = run_random_forest(df_train, df_test, features, target)
    results['XGBoost'] = run_xgboost(df_train, df_test, features, target)
    results['GPR'] = run_gpr(df_train, df_test, features, target)

    # Print Results -----------------------------------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    df_results.index = ['Train_RMSE', 'Train_PctErr', 'Test_RMSE', 'Test_PctErr']
    df_results = df_results.T
    print(df_results)


if __name__ == "__main__":
    main()
