"""
Computational Neurodynamics
Exercise 1

Solves the ODE dy/dt=y (exact solution: y(t)=exp(t)), by numerical
simulation using the Euler method, for two different step sizes.

(C) Murray Shanahan et al, 2016
"""

import numpy as np
import matplotlib.pyplot as plt
import math
# dv/dt = 0.04v^2 + 5v + 140 - u + I
# du/dt = a * (b * v - u)
# if v >> 30 : v = c, u = u + d
I = 10
a = 0.02
b = 0.2
c = -50
d = 2
dt = 0.01      # Step size for exact solution

# Create time points
Tmin = 0
Tmax = 1000
T = np.arange(Tmin, Tmax+dt, dt)
v = np.zeros(len(T))
u = np.zeros(len(T))

# Approximated solution with small integration Step
v[0] = -65  # Initial value
u[0] = -1
for t in xrange(1, len(T)):
    v[t] = v[t-1] + dt * (0.04 * math.pow(v[t-1], 2) + 5*v[t-1] + 140 - u[t-1] + I)
    u[t] = u[t-1] + dt * a * (b * v[t] - u[t-1])
    if v[t] >= 30:
        v[t] = c
        u[t] += d

# Plot the results
plt.plot(T, v, 'b', label='Solution of y')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc=0)
plt.show()
