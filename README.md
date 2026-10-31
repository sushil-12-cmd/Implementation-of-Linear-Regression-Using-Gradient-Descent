# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Start the program.
2. Import the required libraries such as NumPy, Pandas, Matplotlib and Scikit-learn.
3. Load the 50 Startups dataset from the CSV file.
4. Display the dataset and encode the categorical State column into numerical values.
5. Separate the independent variables (features) and dependent variable (Profit).
6. Perform feature scaling using StandardScaler.
7. Initialize the model parameters (theta) and learning rate.
8. Calculate the predicted values and cost function.
9. Apply Gradient Descent to update the model parameters iteratively until convergence.
10. Plot the cost function to visualize the reduction in error.
11. Predict the profit for a new startup using the trained model.
12. Display the predicted profit.
13. Stop the program.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Sushil Shiva R
RegisterNumber: 212224250017

import numpy as np
import matplotlib.pyplot as plt

# Input data
X = np.array([10000, 20000, 30000, 40000, 50000,
              60000, 70000, 80000, 90000, 100000], dtype=float)

y = np.array([15000, 25000, 35000, 45000, 55000,
              65000, 75000, 85000, 95000, 105000], dtype=float)

# Normalize input
X_mean = np.mean(X)
X_std = np.std(X)
X_scaled = (X - X_mean) / X_std

# Initialize parameters
m = 0.0
b = 0.0
alpha = 0.01
iterations = 1000
n = len(X)

# Gradient Descent
cost_history = []

for i in range(iterations):
    y_pred = m * X_scaled + b

    dm = (-2 / n) * np.sum(X_scaled * (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)

    m = m - alpha * dm
    b = b - alpha * db

    cost = np.mean((y - (m * X_scaled + b)) ** 2)
    cost_history.append(cost)

# Display results
print("Slope:", m / X_std)
print("Intercept:", b - (m * X_mean / X_std))
print("Final Cost:", cost_history[-1])

# Prediction
new_spend = 75000
new_scaled = (new_spend - X_mean) / X_std
predicted_profit = m * new_scaled + b

print("R&D Spend:", new_spend)
print("Predicted Profit:", predicted_profit)

# Regression graph
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, m * X_scaled + b, color='red',
         label='Regression Line')
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()

# Cost graph
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.show()

```

## Output:
<img width="905" height="1171" alt="image" src="https://github.com/user-attachments/assets/870bc3b0-d561-458c-95d2-49d8683404a3" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
