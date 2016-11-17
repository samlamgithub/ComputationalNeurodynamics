"""
Computational Neurodynamics
Exercise 3

(C) Murray Shanahan et al, 2015
"""

import numpy as np
import numpy.random as rn
from NetworkRingLattice import NetworkRingLattice


def NetworkWattsStrogatz(N, k, p, w):
  """
  Creates a ring lattice with N nodes and neighbourhood size k, then
  rewires it according to the Watts-Strogatz procedure with probability p.

  Inputs:
  N -- Number of nodes
  k -- Neighbourhood size of the initial ring lattice
  p -- Rewiring probability
  """

  # Create a regular string lattice
  CIJ = NetworkRingLattice(N, k, w)

  # Loop over all connections and swap each of them with probability p
  for i in range(N):
    for j in range(N):
      if CIJ[i, j] and rn.random() < p:
        if i != j:
            # We modify connections in both directions (i.e. [i,j] and [j,i])
            # to maintain network undirectedness (i.e. symmetry).
            CIJ[i, j] = 0
            CIJ[j, i] = 0
            # PEDRO
            # h = np.mod(i + np.ceil(rn.random()*(N-1)) - 1, N)
            h = int(np.mod(i + np.ceil(rn.random()*(N-1)) - 1, N))
            CIJ[i, h] = w()
            # CIJ[h, i] = 1

  return(CIJ)
