import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equation dy/dx = f(x, y)
def f(x, y):
    return 2 * y - y ** 2  # Modify this function as needed

# Define the differential equation for scipy.solve_ivp in the form required
def dydx(t, y):
    return f(t, y)

# Solve the ODE for t >= 0 with initial condition y(0) = 1
initial_condition = [1]
t_span_positive = (0, 3)  # Positive time span
t_eval_positive = np.linspace(0, 3, 100)  # Points at which to evaluate the solution

# Solve the ODE numerically for t >= 0
sol_positive = solve_ivp(dydx, t_span_positive, initial_condition, t_eval=t_eval_positive)

# Solve the ODE for t <= 0 with the same initial condition y(0) = 1
t_span_negative = (0, -3)  # Negative time span
t_eval_negative = np.linspace(0, -3, 100)  # Points at which to evaluate the solution

# Solve the ODE numerically for t <= 0
sol_negative = solve_ivp(dydx, t_span_negative, initial_condition, t_eval=t_eval_negative)

# Set the grid range for x and y
x_min, x_max = -3, 3
y_min, y_max = -3, 3

# Create a grid of points for the direction field
x = np.linspace(x_min, x_max, 20)
y = np.linspace(y_min, y_max, 20)
X, Y = np.meshgrid(x, y)

# Compute dy/dx for each point on the grid
U = np.ones_like(X)  # The x-components (we assume all slopes have an x-component of 1)
V = f(X, Y)  # The y-components are based on the differential equation

# Normalize the arrows to make them visually consistent
N = np.sqrt(U**2 + V**2)
U, V = U / N, V / N

# Plot the direction field using quiver
plt.quiver(X, Y, U, V, color='blue')

# Plot the solution passing through (0, 1) for both positive and negative t
plt.plot(sol_positive.t, sol_positive.y[0], 'r')  # Red color for the solution
plt.plot(sol_negative.t, sol_negative.y[0], 'r')  # Same color for the solution

# Labeling the plot
plt.title(r"Direction Field for $\frac{dy}{dx} = 2y - y^2$ and Solution through $(0,1)$")
plt.xlim([x_min, x_max])
plt.ylim([y_min, y_max])
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('direction_field_solution.png', dpi=300)  # Save the figure as PNG
# You can also save as PDF: plt.savefig('direction_field_solution_no_legend.pdf', dpi=300)

plt.show()
