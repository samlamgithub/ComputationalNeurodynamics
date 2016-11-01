"""
Computational Neurodynamics
Exercise 2

Run an example network with two layers of Izhikevich neurons.

(C) Murray Shanahan et al, 2016
"""
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as pl
import IzNetwork as iz

def ConnectLayers(N1, N2):
  """
  Create a network with two layers of excitatory Izhikevich neurons with N1
  and N2 neurons, respectively.
  """
  N = N1 + N2
  Dmax = 5  # Max synaptic delay, in ms
  net = iz.IzNetwork(N, Dmax)

  F = 50.0/np.sqrt(N1)

  # Build network as a block matrix. Block [i,j] is the connection from
  # layer i to layer j
  W = np.bmat([[np.zeros((N1,N1)), F*np.ones((N1,N2))],
               [np.zeros((N2,N1)),  np.zeros((N2,N2))]])
  D = Dmax*np.ones((N,N), dtype=int)

  # All neurons are heterogeneous excitatory regular spiking
  r = rn.rand(N)
  a = 0.02*np.ones(N)
  b = 0.2*np.ones(N)
  c = -65 + 15*(r**2)
  d = 8 - 6*(r**2)

  # Inhibitory
  # a = 0.02*np.ones(N)
  # b = 0.25*np.ones(N)
  # c = -65 * r
  # d = 2 * r

  # Bursting
  # a = 0.02 * np.ones(N)
  # b = 0.2 * np.ones(N)
  # c = -50 * r
  # d = 2 * r

  net.setWeights(W)
  net.setDelays(D)
  net.setParameters(a, b, c, d)

  return net


if __name__ == '__main__':
  """
  Create and run the network with constant input.
  """
  Tmin = 0
  Tmax = 400

  # Construct the network
  N1 = 4
  N2 = 4
  N  = N1 + N2
  net = ConnectLayers(N1, N2)

  # Set current and initialise arrays
  I = np.hstack([5*np.ones(N1), np.zeros(N2)])
  T = np.arange(Tmin, Tmax + 1)
  V = np.zeros((len(T), N))
  R = np.zeros(len(T))

  # Simulate
  for t in xrange(len(T)):
    net.setCurrent(I)
    fIdx = net.update()
    R[t] = len(fIdx)
    V[t,:],_ = net.getState()

  # Plot results
  pl.figure()

  ax1 = pl.subplot(311)
  ax1.plot(T, R)
  ax1.set_ylabel('number of firing neurons')
  ax1.set_title('raster')

  ax1 = pl.subplot(312)
  ax1.plot(T, V[:, 0:N1])
  ax1.set_ylabel('Voltage (mV)')
  ax1.set_title('Layer 1')

  ax2 = pl.subplot(313)
  ax2.plot(T, V[:, (N1+1):N])
  ax2.set_xlabel('Time (ms)')
  ax2.set_ylabel('Voltage (mV)')
  ax2.set_title('Layer 2')

  pl.show()
