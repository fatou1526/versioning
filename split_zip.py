"""
This module contains all method to split the data and to train ML model

"""
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile

def split_dataset(X, y):
    # This method helps to split the data to train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test


def zipping(zippath, csvpath):
    # This method considers a zip path and a csv path to zip the csv file
    with zipfile.ZipFile(zippath, "w") as zipf:
        zipf.write(csvpath)
    return 0
