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

def ConnectLayers():
  """
  Create a network with 800 excitatory Izhikevich neurons with 200
  Inhibitory neurons, respectively.
  """
  p = 0.0
  moduleNum = 8
  ExiNumPerMod = 100
  ExiModConnNum = 1000
  InhiNum = 200
  N = moduleNum * ExiNumPerMod + InhiNum
  Dmax = 20  # Max synaptic delay, in ms
  net = iz.IzNetwork(N, Dmax)

  EtoESF = 17
  EtoISF = 50
  ItoESF = 2
  ItoISF = 1

  EtoEW = lambda:1
  EtoIW = lambda:np.random.rand()
  ItoEW = lambda:-np.random.rand()
  ItoIW = lambda:-np.random.rand()

  EtoECD = lambda:np.random.randint(1, 20, dtype="int")
  EtoICD = lambda:1
  ItoECD = lambda:1
  ItoICD = lambda:1

  Modules = np.array([])
  EtoEMatrix = np.zeros([ExiNumPerMod*moduleNum, ExiNumPerMod*moduleNum])
  for i in range(moduleNum):
      A = NetworkWattsStrogatz(ExiNumPerMod, ExiModConnNum, p, EtoIW)
      Modules.append(A)
      EtoEMatrix[i*ExiNumPerMod: , i*ExiNumPerMod: ] = A

  EtoIMatrix = np.zeros([ExiNumPerMod*moduleNum, InhiNum])
  EtoIs = []
  for i in range(4):
      A = np.fromfunction(lambda i, j: EtoIW if i==j else 0.0, (InhiNum, InhiNum), dtype=double)
      EtoIs.append(A)

  EtoIMatrix = np.bmat([[EtoIs[0]],[EtoIs[1]],[EtoIs[2]],[EtoIs[3]]])

  ItoEMatrix = np.fromfunction(ItoEW, (InhiNum, ExiNumPerMod*moduleNum), dtype=double)
  ItoIMatrix = np.fromfunction(ItoIEW, (InhiNum, InhiNum), dtype=double)

  # Build network as a block matrix. Block [i,j] is the connection from
  # layer i to layer j
  W = np.bmat([[EtoESF * EtoEMatrix, EtoISF * EtoIMatrix],
               [ItoESF * ItoEMatrix, ItoISF * ItoIMatrix]])

  EtoECDMatrix = np.fromfunction(EtoECD, (ExiNumPerMod*moduleNum, ExiNumPerMod*moduleNum), dtype="double")

  D = np.ones((1000,1000), dtype="double")
  D[0:800,0:800] = ExiNumPerMod*moduleNum

  # All neurons are heterogeneous excitatory regular spiking
  As = []
  Bs = []
  Cs = []
  Ds = []
  for i in range(1000):
      r = rn.rand(N)
      if i < 800:
        # Exi
        Ea = 0.02*np.ones(N)
        Eb = 0.2*np.ones(N)
        Ec = -65 + 15*(r**2)
        Ed = 8 - 6*(r**2)
        As.append(Ea)
        Bs.append(Eb)
        Cs.append(Ec)
        Ds.append(Ed)
      else:
        # Inhibitory
        Ia = 0.02*np.ones(N)
        Ib = 0.25*np.ones(N)
        Ic = -65 * r
        Id = 2 * r
        As.append(Ia)
        Bs.append(Ib)
        Cs.append(Ic)
        Ds.append(Id)

  # Bursting
  # a = 0.02 * np.ones(N)
  # b = 0.2 * np.ones(N)
  # c = -50 * r
  # d = 2 * r

  net.setWeights(W)
  net.setDelays(D)
  net.setParameters(As, Bs, Cs, Ds)

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
