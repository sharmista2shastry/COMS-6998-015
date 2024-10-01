import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# define f(x)
def f(x):
    return x + np.sin(1.5 * x)

# Generate 20 random x values
np.random.seed(42)
x = np.random.uniform(0, 10, 20)

# Generate noise from N(0, 0.3)
epsilon = np.random.normal(0, np.sqrt(0.3), 20)

# Define y
y = f(x) + epsilon

# Fit polynomials of degree 1, 3 and 10
p1 = np.polyfit(x, y, 1)
p3 = np.polyfit(x, y, 3)
p10 = np.polyfit(x, y, 10)

# Print the coefficients
print("Coefficients for Degree 1 Polynomial:", p1)
print("Coefficients for Degree 3 Polynomial:", p3)
print("Coefficients for Degree 10 Polynomial:", p10)

# Generate a smooth x range for plotting
x_smooth = np.linspace(0, 10, 100)

# Evaluate the polynomial models on the smooth x range
y1 = np.polyval(p1, x_smooth)
y3 = np.polyval(p3, x_smooth)
y10 = np.polyval(p10, x_smooth)

# Plot the data and the fitted curves
plt.scatter(x, y, label="y(x) = f(x) + noise", color="blue")  # Scatter plot for the dataset
plt.plot(x_smooth, f(x_smooth), label="f(x) = x + sin(1.5x)", color="red")  # True function f(x)
plt.plot(x_smooth, y1, label="g1(x) - Degree 1", color="green")  # Degree 1 polynomial
plt.plot(x_smooth, y3, label="g3(x) - Degree 3", color="orange")  # Degree 3 polynomial
plt.plot(x_smooth, y10, label="g10(x) - Degree 10", color="purple")  # Degree 10 polynomial

# Add labels and legends
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Fit Comparison (Degree 1, 3, 10)')
plt.show()