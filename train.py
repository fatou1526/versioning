"""
This module allows to encode  the label, to normalize the features, to train the random forest model and to evaluate the model
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# label encoder
def encoding(label):
    le= LabelEncoder()
    label =le.fit_transform(label)
    return label

# Normalization/Standardisation
def normalize(features):
    features = StandardScaler().fit_transform(features)
    return features

# Training
def training(X_train, y_train, X_test):    
    rfc = RandomForestClassifier(n_estimators=100, random_state=24)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    return y_pred

# Evaluate model
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report



