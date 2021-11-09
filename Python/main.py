import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn import metrics
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

#Pull the data
data = pd.read_csv("../Dataset/USA_Housing.csv", sep=",")
data = data[["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms", "Area Population", "Price"]]
predict = "Price"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

'''best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    #Declare Linear Regression Method
    linear = linear_model.LinearRegression()

    #Declare the models
    linear.fit(x_train, y_train)
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best:
        best = accuracy
        with open("housePredictionModel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("../Model/housePredictionModel.pickle", "rb")
linear = pickle.load(pickle_in)


#Getting the value of m and b from y =  mx + b
print("Coefficient", linear.coef_)
print("Intercept", linear.intercept_)

#Getting the predictions
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#Getting the metrics
print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
print('MSE: ', metrics.mean_squared_error(y_test,predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#Plotting the data
x1 = "Avg. Area Income"
x2 = "Area Population"
style.use("ggplot")
pyplot.scatter(data[x2], data["Price"])
pyplot.xlabel(x2)
pyplot.ylabel("Price")
pyplot.show()
