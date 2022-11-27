import numpy as np


def clean_dataframe(df):
    X_df = keep_columns(df, [f'emg'])
    X = X_df.to_numpy()
    y_df = keep_columns(df, ['gt'])
    y = y_df.to_numpy().squeeze()
    return X, y


def drop_columns(df, tuple_of_columns):
    """
    Given a dataframe, and a tuple of column names, this function will search
    through the dataframe and drop the columns which contain a string from the
    list of the undesired columns. All other columns are kept
    """
    if len(tuple_of_columns) >= 1:
        cols = df.columns[df.columns.to_series().str.contains('|'.join(tuple_of_columns))]
        return df.drop(columns=cols)
    return df


def keep_columns(df, tuple_of_columns):
    """
    Given a dataframe, and a tuple of column names, this function will search
    through the dataframe and keep only columns which contain a string from the
    list of the desired columns. All other columns are removed
    """
    if len(tuple_of_columns) >= 1:
        cols = df.columns[df.columns.to_series().str.contains('|'.join(tuple_of_columns))]
        return df[cols]
    return df


def random_split(X, y, split=0.8):
    assert len(X) == len(y)
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    num_train_samples = round(num_samples * split)
    X_train, X_test = X[indices[:num_train_samples]], X[indices[num_train_samples:]]
    y_train, y_test = y[indices[:num_train_samples]], y[indices[num_train_samples:]]
    return X_train, y_train, X_test, y_test