from dataset.dataset_preprocessing import *
from models.m1_linear_regression import *
from models.m2_svr import *
from models.m3_random_forest import *
from models.m4_xgboost import *
from models.m5_gpr import *


def main():
    # Load and process data
    dataset_path = './dataset/boston_housing_prices.csv'
    # dataset_statistics(dataset_path=dataset_path)
    df_train, df_test, features, target = preprocess_data(dataset_path=dataset_path, process='abs_scaling')

    # Run Models
    results = {}
    results['Linear_Regression'] = run_linear_regression(df_train, df_test, features, target)
    results['SVR'] = run_svr(df_train, df_test, features, target)
    results['Random_Forest'] = run_random_forest(df_train, df_test, features, target)
    results['XGBoost'] = run_xgboost(df_train, df_test, features, target)
    results['GPR'] = run_gpr(df_train, df_test, features, target)

    # Print Results
    df_results = pd.DataFrame(results)
    df_results.index = ['Train_RMSE', 'Train_PctErr', 'Test_RMSE', 'Test_PctErr']
    print(df_results)


if __name__ == "__main__":
    main()
