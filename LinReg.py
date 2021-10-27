import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = fetch_california_housing()
X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

sklearn_regressor = LinearRegression().fit(X_train, Y_train)
sklearn_train_accuracy = sklearn_regressor.score(X_train, Y_train)
sklearn_test_accuracy = sklearn_regressor.score(X_test, Y_test)

print(sklearn_train_accuracy, sklearn_test_accuracy)