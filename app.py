from preprocess_module import load_data
from preprocess_module import preprocessing_duplicates
from preprocess_module import preprocessing_outliers
from preprocess_module import preprocessing_null
from split_zip import split_dataset
from split_zip import zipping
# Loading data
data = load_data('Crop_recommendation.csv')
print (data.head())

# Preprocessing data (duplicates, outliers, null )
data = preprocessing_duplicates(data)
data = preprocessing_outliers(data)
data = preprocessing_null(data)

# Splitting data
y = data['label']
X = data.drop('label', axis=1)
X_train, X_test, y_train, y_test = split_dataset(X, y)

# Converting the splitted datasets to csv
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Zipping the datasets
zipping('X_train.zip','X_train.csv')
zipping('X_test.zip','X_test.csv')
zipping('y_train.zip','y_train.csv')
zipping('y_test.zip','y_test.csv')



