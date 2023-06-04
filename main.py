import pandas as pd
import numpy as np
from sklearn import model_selection
from models.all_models import *


def main():
    df = pd.read_csv('./dataset/boston_housing_prices.csv')
    df_train, df_test = model_selection.train_test_split(df, test_size=0.2)
    features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    target = 'MEDV'
    answer = gpr(df_train, df_test, features, target)
    print(answer)


if __name__ == "__main__":
    main()
