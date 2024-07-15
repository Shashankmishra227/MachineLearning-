import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv("datalr.csv")

X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Split the data into train, test, and validation sets
train_size = int(0.7 * len(data))
test_size = int(0.15 * len(data))
val_size = len(data) - train_size - test_size

X_train = X[:train_size]
Y_train = Y[:train_size]

X_test = X[train_size:train_size+test_size]
Y_test = Y[train_size:train_size+test_size]

X_val = X[train_size + test_size:]
Y_val = Y[train_size + test_size:]
# print(data.iloc[0:, 1])

class SimpleLinearRegration:
    def __init__(self, learning_rate=0.0001, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.intercept = 0
        self.slope = 0

    def fit(self, X, Y):
        n = len(Y)
        self.intercept = 0
        self.slope = 0

        for i in range(self.iterations):
            Y_pred = self.slope*X + self.intercept
            error = Y_pred - Y

            d_intercept = -(1/n)*np.sum(error)
            d_slope = -(1/n)*np.sum(error * X)
            
            self.slope += self.learning_rate*d_slope
            self.intercept += self.learning_rate*d_intercept

            mse = (1/n)*np.sum(error**2)

            print(f"itrations {i+1} Mean Square Error {mse}")

    def predict(self, X):
        return (self.intercept + self.slope*X)
 

model = SimpleLinearRegration()
model.fit(X_train, Y_train)

# Predict values for train, test, and validation sets

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_val_pred = model.predict(X_val)

# Plot the results
plt.figure(figsize=(14, 7))

# Training set results
plt.subplot(1, 3, 1)
plt.scatter(X_train, Y_train, color='blue', label='Actual')
plt.plot(X_train, y_train_pred, color='red', label='Predicted')
plt.title('Training set')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Testing set results
plt.subplot(1, 3, 2)
plt.scatter(X_test, Y_test, color='blue', label='Actual')
plt.plot(X_test, y_test_pred, color='red', label='Predicted')
plt.title('Testing set')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Validation set results
plt.subplot(1, 3, 3)
plt.scatter(X_val, Y_val, color='blue', label='Actual')
plt.plot(X_val, y_val_pred, color='red', label='Predicted')
plt.title('Validation set')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()