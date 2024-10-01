import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# define base function
def f(x):
    return x + np.sin(1.5 * x)

# define epsilon error function
def epsilon(n):
    return np.random.normal(loc=0, scale=0.3, size=n)

# define estimate function
def y(x):
    return f(x) + epsilon(len(x))

# Parameters
n_observations_per_dataset = 50
n_datasets = 100
max_poly_degree = 15
model_poly_degrees = range(1, max_poly_degree + 1)
n_train = int(np.ceil(n_observations_per_dataset * 0.8))
x_range = 10

# Storage for errors and predictions
train_errors = defaultdict(list)
test_errors = defaultdict(list)
pred_test = defaultdict(list)

# Generate datasets and fit polynomials of different degrees
for dataset in range(n_datasets):
    
    # Generate random data points and split into train/test
    x = np.random.random_sample(n_observations_per_dataset) * x_range
    x_train, x_test = x[:n_train], x[n_train:]
    
    # Simulated target outputs for training and testing sets
    y_train = y(x_train)
    y_test = y(x_test)

    # Fit models for each polynomial degree
    for degree in model_poly_degrees:
        theta_hat = np.polyfit(x_train, y_train, degree)
        
        # Predictions for train and test sets
        y_train_pred = np.polyval(theta_hat, x_train)
        y_test_pred = np.polyval(theta_hat, x_test)

        # Store mean squared errors
        train_errors[degree].append(np.mean((y_train_pred - y_train) ** 2))
        test_errors[degree].append(np.mean((y_test_pred - y_test) ** 2))

        # Store predictions for bias/variance calculation
        pred_test[degree].append(y_test_pred)

# Functions to calculate bias and variance
def calculate_bias_squared(pred_test, true_y):
    avg_pred = np.mean(pred_test, axis=0)  # E[g(x)]
    return np.mean((avg_pred - true_y) ** 2)

def calculate_variance(pred_test):
    avg_pred = np.mean(pred_test, axis=0)  # E[g(x)]
    return np.mean((pred_test - avg_pred) ** 2)

# Bias-Variance Decomposition and Error Aggregation
bias_squared = []
variance = []
complexity_train_error = []
complexity_test_error = []

for degree in model_poly_degrees:
    pred_test_array = np.array(pred_test[degree])
    complexity_train_error.append(np.mean(train_errors[degree]))
    complexity_test_error.append(np.mean(test_errors[degree]))
    bias_squared.append(calculate_bias_squared(pred_test_array, y_test))
    variance.append(calculate_variance(pred_test_array))

# Identify the best model based on test error
best_model_degree = model_poly_degrees[np.argmin(complexity_test_error)]
print(f"The best model is degree {best_model_degree}")

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(14, 10))

# Plot Bias^2, Variance, and Testing Error
axs[0].plot(model_poly_degrees, bias_squared, label='Bias^2', color='blue')
axs[0].plot(model_poly_degrees, variance, label='Variance', color='green')
axs[0].plot(model_poly_degrees, np.array(bias_squared) + np.array(variance), label='Bias^2 + Variance', color='gray', linestyle='-.')
axs[0].plot(model_poly_degrees, complexity_test_error, label='Testing Error', color='red', linewidth=3)
axs[0].axvline(best_model_degree, linestyle='--', color='black', label=f'Best Model (degree={best_model_degree})')
axs[0].axhline(0.09, color='tomato', linestyle='--', label='$\sigma^2$')
axs[0].set_xlabel('Model Complexity (Polynomial Degree)')
axs[0].set_yscale('log')
axs[0].set_title('Bias-Variance Tradeoff and Testing Error')
axs[0].legend()

# Plot Training and Testing Errors
axs[1].plot(model_poly_degrees, complexity_train_error, label='Training Error', color='black', linewidth=3)
axs[1].plot(model_poly_degrees, complexity_test_error, label='Testing Error', color='red', linewidth=3)
axs[1].axvline(best_model_degree, linestyle='--', color='black', label=f'Best Model (degree={best_model_degree})')
axs[1].set_xlabel('Model Complexity (Polynomial Degree)')
axs[1].set_yscale('log')
axs[1].set_title('Training and Testing Errors')
axs[1].legend(loc='upper center')

plt.tight_layout()
plt.show()