import misc_utils as mu
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv('collected_data/2022_09_19/dm_21.csv', index_col=0)
    X, y = mu.clean_dataframe(df)
    X_train, y_train, X_test, y_test = mu.random_split(X, y)
    print(f'training data shape: {X_train.shape}, testing data shape: {X_test.shape}')