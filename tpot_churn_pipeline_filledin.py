import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from tpot.builtins import ZeroCount
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
# Setting the target column to be 'Churn':

tpot_data = pd.read_csv('../data/cleaned_churn_data.csv', index_col='customerID')
features = tpot_data.drop('Churn', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Churn'], random_state=42)

# Average CV score on the training set was: 0.7985568791032367
exported_pipeline = make_pipeline(
    RobustScaler(),
    ZeroCount(),
    MLPClassifier(alpha=0.0001, learning_rate_init=0.01)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

# Adding a line to output the results
print(results)
