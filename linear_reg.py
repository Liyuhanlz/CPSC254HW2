import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Loading Dataset

X = np.array([-3.0, -2.5, -2.0, -1.5, -1.0,
              0.0, 1.0, 1.5, 2.0, 2.5, 2.7])

Y = np.array([15.5, 12.9, 9.5, 6.2, 5.8,
              5.5, 7.1, 9.7, 13.5, 18.4, 21.4])

X_reshaped = X.reshape(-1, 1)


# Linear Regression

model = LinearRegression()
model.fit(X_reshaped, Y)

# Displaying Coefficients

w0 = model.intercept_
w1 = model.coef_[0]

print("Coefficient Vector:")
print([w0, w1])

print("\nRegression Equation:")
print(f"y = {w0:.4f} + ({w1:.4f})x")

# RMSE

Y_pred = model.predict(X_reshaped)

rmse = np.sqrt(mean_squared_error(Y, Y_pred))

print(f"\nTraining RMSE = {rmse:.4f}")

# Scatter plot

plt.scatter(X, Y, color='blue', label='Data')

plt.plot(
    X,
    Y_pred,
    color='red',
    linewidth=2,
    label='Regression Line'
)

plt.title("Training Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# GasProperties.csv

df = pd.read_csv("GasProperties.csv")

X = df[['T', 'P', 'TC', 'SV']]
y = df['Idx']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Least Squares Regression

# Converting DataFrames to NumPy arrays
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Adding bias column (w0)
X_train_bias = np.c_[np.ones(X_train_np.shape[0]), X_train_np]
X_test_bias = np.c_[np.ones(X_test_np.shape[0]), X_test_np]

# Least Squares Formula
w = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train_np

print("\nCoefficient Vector:")
print(w)

# Training predictions
y_train_pred = X_train_bias @ w

# Testing predictions
y_test_pred = X_test_bias @ w

# Training RMSE
train_rmse = np.sqrt(
    np.mean((y_train_np - y_train_pred) ** 2)
)

# Testing RMSE
test_rmse = np.sqrt(
    np.mean((y_test_np - y_test_pred) ** 2)
)

print(f"\nTraining RMSE: {train_rmse:.6f}")
print(f"Testing RMSE: {test_rmse:.6f}")

print("\nLeast Squares Equation:")
print(
    f"Idx = {w[0]:.4f} "
    f"+ ({w[1]:.4f})T "
    f"+ ({w[2]:.4f})P "
    f"+ ({w[3]:.4f})TC "
    f"+ ({w[4]:.4f})SV"
)
