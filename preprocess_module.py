"""
This module contains methods to preprocess the data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def laod_data(filepath):
    # loading dataset
    df = pd.read_csv(filepath)
    return df
