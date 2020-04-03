from sklearn.linear_model import LinearRegression

import Data
#Mean Absolute Error (MAE): MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
# It's the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight
from sklearn.metrics import mean_absolute_error
#scikit-learn provides a helpful function for partitioning data, train_test_split , which splits out your data into a training set and a test set. ...
# We provide the proportion of data to use as a test set and we can provide the parameter random_state , which is a seed to ensure repeatable results
from sklearn.model_selection import train_test_split
#Linear regression quantifies the relationship between one or more predictor variables and one outcome variable
from sklearn.linear_model import LinearRegression

from Data import X, y

train_X, test_X, train_y, test_y = train_test_split(X, y,test_size=0.2,random_state = 1)
Training_Model= LinearRegression()
Training_Model.fit(train_X,train_y)
train_predictions=Training_Model.predict(train_X)

print("Training Error using 80% of training data is ",mean_absolute_error(train_predictions,train_y))
test_predictions=Training_Model.predict(test_X)
#print(test_predictions)
print("Testing error using 20% of testing data is",mean_absolute_error(test_predictions,test_y))
A= mean_absolute_error(train_predictions,train_y)
B= mean_absolute_error(test_predictions,test_y)

Accuracy =(A/B)*100
print("Accuracy is ",Accuracy)
