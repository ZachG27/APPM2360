import numpy as np
import matplotlib.pyplot as plt

# Define the function for the slope field (y' = 2y - y^2)
def slope_field(y):
    return 2 * y - y**2

# Create a grid of points (t, y)
t_vals = np.linspace(-3, 3, 20)
y_vals = np.linspace(-3, 3, 20)
T, Y = np.meshgrid(t_vals, y_vals)

# Compute the slope at each point
dY = slope_field(Y)
dT = np.ones_like(dY)  # For the horizontal axis, change in t is always 1

# Normalize the arrows for better visualization
N = np.sqrt(dT**2 + dY**2)
dT /= N
dY /= N

# Create the plot
plt.figure(figsize=(8, 6))
plt.quiver(T, Y, dT, dY, angles='xy')

# Plot the solution curve (example: solution passing through (0,1))
t_sol = np.linspace(-3, 3, 100)
y_sol = 2 / (1 + np.exp(-2 * t_sol))  # Analytical solution to y' = 2y - y^2
plt.plot(t_sol, y_sol, 'r', label="Solution curve through (0,1)")

# Add labels and title
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.title("Direction Field for $y' = 2y - y^2$")
plt.xlabel("$t$")
plt.ylabel("$y$")
plt.legend()
plt.grid()

# Save the plot as a PNG file
plt.savefig("direction_field.png", dpi=300, bbox_inches='tight')
plt.show()
