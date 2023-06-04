import pandas as pd
import numpy as np
from dataset.dataset_preprocessing import *
from models.all_models import *


def main():
    dataset_path = './dataset/boston_housing_prices.csv'
    # dataset_statistics(dataset_path=dataset_path)
    df_train, df_test, features, target = preprocess_data(dataset_path=dataset_path, process='abs_scaling')
    # answer = gpr(df_train, df_test, features, target)
    # print(answer)


if __name__ == "__main__":
    main()
