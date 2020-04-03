import pandas as pd
import sklearn
import scipy
import numpy

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats


path = 'C:\\Users\\user\\Downloads\\melb_data.csv'

DataWithNull = pd.read_csv(path)
#Gives Information about the columns
DataWithNull.columns
#the shape of an array is a tuple of integers giving the size of the array along each dimension
DataWithNull.shape
#Pandas Series.isnull() function detect missing values in the given series object.
#It return a boolean same-sized object indicating if the values are NA. Missing values gets mapped to True and non-missing value gets mapped to False .
nullvalues= DataWithNull.isnull().sum()
print(nullvalues)
DataWithNull.describe()
#Dropping the data with null-values
Data=DataWithNull.dropna()
#Set Index for the data
Data=Data.reset_index(drop=True)
#Consider Features as sub-division of our Data-Set
features = ['Rooms','Bathroom', 'Landsize', 'Lattitude', 'Longtitude','Bedroom2','Car','BuildingArea','YearBuilt']
X = Data[features]

X['Rooms'].describe()
X['Bedroom2'].describe()
X['Bathroom'].describe()
X['Landsize'].describe()
X['BuildingArea'].describe()
X['Car'].describe()
X = X[(X.Rooms <= 4) &(X.Bathroom <= 2) & (X.Bedroom2 <=5) & (X.Landsize <= 652) & (X.BuildingArea>=100 ) &(X.Car<=4) ]
Data =Data[(Data.Rooms <= 4) &(Data.Bathroom <= 2) & (Data.Bedroom2 <=5) & (Data.Landsize <= 652) & (Data.BuildingArea>=100) & (Data.Car <=4) ]
# Actual price is stored in y
y= Data.Price
"""Head command can print top  rows in the csv file"""
head=X.head()
# Define model
Model = LinearRegression()
# Fitting the model
Model.fit(X,y)
predicted_home_prices = Model.predict(X)
print("Actual Price ",y)
print("Predicted prices are ",predicted_home_prices)
error=mean_absolute_error(y, predicted_home_prices)
print("Error using 100% training data is", error)