import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.linear_model import Ridge

# Define base function
def f(x):
    return x + np.sin(1.5 * x)

# Define epsilon error function
def epsilon(n):
    return np.random.normal(loc=0, scale=0.3, size=n)

# Define estimate function
def y(x):
    return f(x) + epsilon(len(x))

# Parameters
n_observations_per_dataset = 50
n_datasets = 100
degree = 10  
n_train = int(np.ceil(n_observations_per_dataset * 0.8))
x_range = 10
regularization_strength = 0.01

# Storage for errors and predictions
train_errors = defaultdict(list)
test_errors = defaultdict(list)
pred_test_unreg = []
pred_test_reg = []

# Generate datasets and fit the degree-10 polynomial with and without regularization
for dataset in range(n_datasets):
    
    # Generate random data points and split into train/test
    x = np.random.random_sample(n_observations_per_dataset) * x_range
    x_train, x_test = x[:n_train], x[n_train:]
    
    # Simulated target outputs for training and testing sets
    y_train = y(x_train)
    y_test = y(x_test)

    # Unregularized polynomial fit
    theta_hat_unreg = np.polyfit(x_train, y_train, degree)
    
    # Predictions for unregularized model
    y_train_pred_unreg = np.polyval(theta_hat_unreg, x_train)
    y_test_pred_unreg = np.polyval(theta_hat_unreg, x_test)

    # Store predictions and errors for unregularized model
    pred_test_unreg.append(y_test_pred_unreg)
    train_errors['unregularized'].append(np.mean((y_train_pred_unreg - y_train) ** 2))
    test_errors['unregularized'].append(np.mean((y_test_pred_unreg - y_test) ** 2))

    # Prepare X for Ridge Regression
    X_train_poly = np.vander(x_train, N=degree+1, increasing=True)
    X_test_poly = np.vander(x_test, N=degree+1, increasing=True)

    # Ridge regression (L2 regularization)
    ridge_model = Ridge(alpha=regularization_strength, fit_intercept=False)
    ridge_model.fit(X_train_poly, y_train)

    # Predictions for regularized model
    y_train_pred_reg = ridge_model.predict(X_train_poly)
    y_test_pred_reg = ridge_model.predict(X_test_poly)

    # Store predictions and errors for regularized model
    pred_test_reg.append(y_test_pred_reg)
    train_errors['regularized'].append(np.mean((y_train_pred_reg - y_train) ** 2))
    test_errors['regularized'].append(np.mean((y_test_pred_reg - y_test) ** 2))

# Functions to calculate bias and variance
def calculate_bias_squared(pred_test, true_y):
    avg_pred = np.mean(pred_test, axis=0)  # E[g(x)]
    return np.mean((avg_pred - true_y) ** 2)

def calculate_variance(pred_test):
    avg_pred = np.mean(pred_test, axis=0)  # E[g(x)]
    return np.mean((pred_test - avg_pred) ** 2)

# Bias-Variance Decomposition and Error Aggregation
bias_squared_unreg = calculate_bias_squared(np.array(pred_test_unreg), y_test)
variance_unreg = calculate_variance(np.array(pred_test_unreg))
mse_unreg = np.mean(test_errors['unregularized'])

bias_squared_reg = calculate_bias_squared(np.array(pred_test_reg), y_test)
variance_reg = calculate_variance(np.array(pred_test_reg))
mse_reg = np.mean(test_errors['regularized'])

# Print results
print(f"Unregularized model (degree 10): Bias^2 = {bias_squared_unreg}, Variance = {variance_unreg}, MSE = {mse_unreg}")
print(f"Regularized model (degree 10 with L2): Bias^2 = {bias_squared_reg}, Variance = {variance_reg}, MSE = {mse_reg}")

# Visualization
fig, axs = plt.subplots(1, 1, figsize=(8, 6))

# Plot Bias^2, Variance, and Testing Error for unregularized and regularized models
width = 0.35  
labels = ['Bias^2', 'Variance', 'MSE']
unreg_values = [bias_squared_unreg, variance_unreg, mse_unreg]
reg_values = [bias_squared_reg, variance_reg, mse_reg]

x = np.arange(len(labels)) 
axs.bar(x - width/2, unreg_values, width, label='Unregularized', color='blue')
axs.bar(x + width/2, reg_values, width, label='Regularized (L2)', color='green')

# Set plot labels and title
axs.set_ylabel('Values')
axs.set_title('Bias^2, Variance, and MSE: Unregularized vs Regularized')
axs.set_xticks(x)
axs.set_xticklabels(labels)
axs.legend()

plt.tight_layout()
plt.show()