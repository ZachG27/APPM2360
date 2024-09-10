import numpy as np
import pandas as pd

# Define the differential equation dy/dt = t * sqrt(y)
def f(t, y):
    return t * np.sqrt(y)

# Define the actual solution for comparison
def actual_solution(t):
    return ((t**2 + 7)**2) / 16

# Euler's method implementation
def euler_method(y0, t0, tf, h):
    N = int((tf - t0) / h)
    t = np.linspace(t0, tf, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for i in range(N):
        y[i + 1] = y[i] + h * f(t[i], y[i])
    return pd.DataFrame({'t': t, 'Approximate y': y})

# Parameters
y0 = 4
t0 = 1
tf = 1.5
h1 = 0.1
h2 = 0.05

# Compute results for h1 and h2
results_h1 = euler_method(y0, t0, tf, h1)
results_h2 = euler_method(y0, t0, tf, h2)

# Compute actual values for both time grids
results_h1['Actual y'] = actual_solution(results_h1['t'])
results_h2['Actual y'] = actual_solution(results_h2['t'])

# Interpolate Approximate y from h=0.05 to the h=0.1 grid
interpolated_y_h2 = np.interp(results_h1['t'], results_h2['t'], results_h2['Approximate y'])

# Add interpolated h=0.05 Approximate y to the results_h1 DataFrame
results_h1['Approximate y h=0.05'] = interpolated_y_h2

# Reorder columns as requested: h1, h2, actual, errors
results_h1 = results_h1[['t', 'Approximate y', 'Approximate y h=0.05', 'Actual y']]

# Calculate errors for both h=0.1 and h=0.05
results_h1['Error h=0.1'] = results_h1['Actual y'] - results_h1['Approximate y']
results_h1['Error h=0.05'] = results_h1['Actual y'] - results_h1['Approximate y h=0.05']

# Round results
results_h1 = results_h1.round(6)

# Save to CSV
csv_path = "Euler_Method_Results_Comparison.csv"
results_h1.to_csv(csv_path, index=False)

# Print the results to console
print(results_h1)
print(f"Results saved to {csv_path}")
