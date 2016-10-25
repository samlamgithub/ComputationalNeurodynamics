"""
Computational Neurodynamics
Exercise 1

Solves the ODE dy/dt=y (exact solution: y(t)=exp(t)), by numerical
simulation using the Euler method, for two different step sizes.

(C) Murray Shanahan et al, 2016
"""

import numpy as np
import matplotlib.pyplot as plt

# y(0) = 1
# y'(0) = 0
m = 1
c = 0.1
k = 1

dt = 0.001      # Step size for exact solution

# Create time points
Tmin = 0
Tmax = 100
T = np.arange(Tmin, Tmax+dt, dt)
y = np.zeros(len(T))
z1 = np.zeros(len(T))
z2 = np.zeros(len(T))

# Approximated solution with small integration Step
y[0] = 1  # Initial value
z1[0] = 0
for t in xrange(1, len(T)):
    z2[t] = -1/m * (c * z1[t-1] + k * y[t-1])
    z1[t] = z1[t-1] + dt * z2[t]
    y[t] = y[t-1] + dt * z1[t]

# Plot the results
plt.plot(T, y, 'b', label='Solution of y')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()
