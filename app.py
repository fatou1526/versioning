from preprocess_module import load_data
from preprocess_module import preprocessing_duplicates
from preprocess_module import preprocessing_outliers
from preprocess_module import preprocessing_null

# Loading data
data = load_data('Crop_recommendation.csv')
print(data.head())

# Preprocessing data (duplicates, outliers, null )
data = preprocessing_duplicates(data)
data = preprocessing_outliers(data)
data = preprocessing_null(data)
print(data.head())


