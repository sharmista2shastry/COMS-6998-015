import numpy as np
import matplotlib.pyplot as plt

# Define f(x) = x + sin(1.5x)
def f(x):
    return x + np.sin(1.5 * x)

# Generate 20 random x values
np.random.seed(42)
x = np.random.uniform(0, 10, 20)

# Generate noise from N(0, 0.3)
epsilon = np.random.normal(0, np.sqrt(0.3), 20)

# Create y(x) = f(x) + epsilon
y = f(x) + epsilon

# Scatterplot for y
plt.scatter(x, y, label="y(x) = f(x) + noise", color="blue")

# Plot a smooth line for f(x)
x_smooth = np.linspace(0, 10, 100)
plt.plot(x_smooth, f(x_smooth), label="f(x) = x + sin(1.5x)", color="red")

# Add labels and legends
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Scatter plot of y(x) and smooth line plot of f(x)')
plt.show()