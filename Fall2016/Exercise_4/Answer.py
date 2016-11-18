"""
Computational Neurodynamics
Exercise 4

Jiahao Lin 2016
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rn
# from IzNetwork import IzNetwork
from NetworkWattsStrogatz import NetworkWattsStrogatz
import IzNetwork as iz

def BackgroundNoise():
    lam = 0.01
    samples = np.random.binomial(1, lam, 1000)
    currents = map(lambda x: 15.0 if x > 0.0 else 0.0, samples)
    return currents

def Network(ExiNumPerMod, ExiModConnNum, EtoIW):
    AA = np.zeros([ExiNumPerMod, ExiNumPerMod])
    connections = rn.randint(0, ExiNumPerMod**2,ExiModConnNum)
    for ii in range(ExiModConnNum):
        vv = connections[ii]
        xindex = vv/ExiNumPerMod
        yindex = vv%ExiNumPerMod
        AA[xindex, yindex] = EtoIW()
    return AA

def ConnectLayers(p):
  """
  Create a network with 800 excitatory Izhikevich neurons with 200
  Inhibitory neurons, respectively.
  """
  moduleNum = 8
  ExiNumPerMod = 100
  ExiModConnNum = 1000
  InhiNum = 200
  N = moduleNum * ExiNumPerMod + InhiNum
  Dmax = 20  # Max synaptic delay, in ms
  net = iz.IzNetwork(N, Dmax)

  EtoESF = 17.0
  EtoISF = 50.0
  ItoESF = 2.0
  ItoISF = 1.0

  EtoEW = lambda:1
  EtoIW = lambda:np.random.rand()
  ItoEW = lambda:-np.random.rand()
  ItoIW = lambda:-np.random.rand()

  EtoECD = lambda i, j:np.random.randint(1, 20, dtype="int")
  EtoICD = lambda:1
  ItoECD = lambda:1
  ItoICD = lambda:1

  Modules = np.zeros([8, 100, 100])
  EtoEMatrix = np.zeros([ExiNumPerMod*moduleNum, ExiNumPerMod*moduleNum])
  for i in range(moduleNum):
      A = Network(ExiNumPerMod, ExiModConnNum, EtoIW)
    #   plt.matshow(A)
    #   plt.show()
    #   print "A shape: ", A.shape
      Modules[i, :, :] = A
      startX = i*ExiNumPerMod
      endX = startX + 100
      EtoEMatrix[startX: endX, startX: endX] = A

  B = EtoEMatrix[0: 800, 0: 800]
  # print "Modules shape: ", Modules.shape
  for i in range(8):
     m = Modules[i]
     for xx in range(100):
        for yy in range(100):
            # print m
            if m[xx, yy] > 0.0:
                # print "pp: ", p
                if np.random.binomial(1, p, 1)[0] > 0.0:
                    # print "rewire"
                    mNum = np.random.randint(0, 7, dtype="int")
                    if mNum == i:
                        mNum = mNum + 1
                    WireNum = np.random.randint(0, 100, dtype="int")
                    B[i*100+xx, i*100+yy] = 0
                    # print ">> ",mNum, WireNum, mNum*800+WireNum
                    B[i*100+xx, mNum*100+WireNum] = 1

  EtoEMatrix[0: 800, 0: 800] = B

  EtoIMatrix = np.zeros([ExiNumPerMod*moduleNum, InhiNum])
  EtoIs = np.zeros([800, 200])
  for i in range(4):
      A = np.zeros([InhiNum, InhiNum])
      for j in range(200):
        A[j, j] = EtoIW()
    #   print("A shap: ", A.shape)
    #   print("f: ", (EtoIs[i*200 :(i+1)*200, 0:200]).shape)
    #   A = np.fromfunction(diagonalRandom, (InhiNum, InhiNum), dtype="double")
      EtoIs[i*200:(i+1)*200, 0:200] = A
  # print("EtoIs shap: ", EtoIs.shape)

  EtoIMatrix = EtoIs

  ItoEMatrix = np.zeros([InhiNum, ExiNumPerMod*moduleNum])
  for i in range(InhiNum):
     for j in range(ExiNumPerMod*moduleNum):
         ItoEMatrix[i, j] = ItoEW()

  ItoIMatrix = np.zeros([InhiNum, InhiNum])
  for i in range(InhiNum):
     for j in range(InhiNum):
         ItoIMatrix[i, j] = ItoIW()

  # Build network as a block matrix. Block [i,j] is the connection from
  # layer i to layer j
  # print "EtoEMatrix shape: ", EtoEMatrix.shape
  # print "EtoIMatrix shape: ", EtoIMatrix.shape
  # print "ItoEMatrix shape: ", ItoEMatrix.shape
  # print "ItoIMatrix shape: ", ItoIMatrix.shape
  W = np.bmat([[EtoESF * EtoEMatrix, EtoISF * EtoIMatrix],
               [ItoESF * ItoEMatrix, ItoISF * ItoIMatrix]])

  plt.matshow(W)
  plt.show()
  # return 0
  # print "w shape: ", W.shape
  EtoECDMatrix = np.ones([1000, 1000], dtype="int")
  for q in range(1000):
      for w in range(1000):
        #   print W[q, w]
          if W[q, w] > 0.0:
              if q < 800 and w < 800:
                  dd = np.random.randint(0, 20, dtype="int")
                  EtoECDMatrix[q, w] = EtoECDMatrix[q, w] + dd
  # EtoECDMatrix = np.fromfunction(EtoECD, (ExiNumPerMod*moduleNum, ExiNumPerMod*moduleNum), dtype="double")
  # print EtoECDMatrix
  D = EtoECDMatrix
  # plt.matshow(D)
  # plt.show()

  # All neurons are heterogeneous excitatory regular spiking
  As = np.zeros([1000, ])
  Bs = np.zeros([1000, ])
  Cs = np.zeros([1000, ])
  Ds = np.zeros([1000, ])
  for i in range(1000):
    #   print is
    #   r = rn.rand()
      if i < 800:
        # Exi
        Ea = 0.02
        Eb = 0.2
        Ec = -65 #+ 15*(r**2)
        Ed = 8 #- 6*(r**2)
        As[i] = Ea
        Bs[i] = Eb
        Cs[i] = Ec
        Ds[i] = Ed
      else:
        # Inhibitory
        Ia = 0.02
        Ib = 0.25
        Ic = -65 #* r
        Id = 2 #* r
        As[i] = Ia
        Bs[i] = Ib
        Cs[i] = Ic
        Ds[i] = Id

  # Bursting
  # a = 0.02 * np.ones(N)
  # b = 0.2 * np.ones(N)
  # c = -50 * r
  # d = 2 * r

  net.setWeights(W)
  net.setDelays(D)
  net.setParameters(np.array(As), np.array(Bs), np.array(Cs), np.array(Ds))

  return net

def main(p):
      """
      Create and run the network with constant input.
      """
    #   print BackgroundNoise()
    #   return 0
    #   print "p: ", p

      Tmin = 0
      Tmax = 1000

      # Construct the network
      net = ConnectLayers(p)
    #   return 0
      # Set current and initialise arrays
      N = 1000
      I = BackgroundNoise()
      T = np.arange(Tmin, Tmax + 1)
      V = np.zeros((len(T), N))
      R = np.zeros(len(T))
      Rx = np.array([])
      Ry = np.array([])
      MeanFireX = [[]] * 8
      MeanFireY = [[]] * 8

      # Simulate
      c1 = 0
      c2 = 0
      for t in xrange(len(T)):
        I = BackgroundNoise()
        net.setCurrent(I)
        fIdx = net.update()
        # print fIdx
        for k in range(len(fIdx)):
            idx = fIdx[k]
            c1 += 1
            if idx < 800:
                c2 +=1
                # print "idx: ", idx
                # print "idx/100: ", idx/100
                # plt.plot([t], [idx])
                Rx = np.append(Rx, t)
                Ry = np.append(Ry, idx)
                MeanFireX[idx/100] = np.append(MeanFireX[idx/100], t)
                MeanFireY[idx/100] = np.append(MeanFireY[idx/100], idx)
            # R[t] = len(fIdx)
            # V[t,:],_ = net.getState()
      print "len: ",  Rx.shape, Ry.shape, c1, c2
    #   plt.plot(Rx, Ry, "o")
    #   plt.show()
    #   return 0
      MeanFireXXs = [[]] * 8
      MeanFireYYs = [[]] * 8
      for mInx in range(8):
          Xc = MeanFireX[mInx]
          Yc = MeanFireY[mInx]
          for windowIndex in range(50):
              startT = 20 * windowIndex
              endT = startT + 50
              countPoint = 0
              for v in range(len(Xc)):
                  if Xc[v] >= startT and Xc[v] < endT:
                      countPoint += 1
              MeanFireXXs[mInx] = np.append(MeanFireXXs[mInx], startT + 25)
              MeanFireYYs[mInx] = np.append(MeanFireYYs[mInx], countPoint)

      # Plot results
      plt.figure()

      ax1 = plt.subplot(211)
      ax1.plot(Rx, Ry, "o")
      ax1.set_ylabel('Neuron number')
      ax1.set_xlabel('Time(ms) + 0s')
      ax1.set_xlim(0, 1000)
      ax1.set_ylim(0, 800)
      ax1.set_title('Raster')

      ax1 = plt.subplot(212)
      for mIdnx in np.arange(8):
          ax1.plot(MeanFireXXs[mIdnx], MeanFireYYs[mIdnx])
      ax1.set_ylabel('Mean firing rate')
      ax1.set_xlim(0, 1000)
    #   ax1.set_ylim(0, 10)
      ax1.set_ylabel('Neuron number')
      ax1.set_title('Time(ms) + 0s')

      plt.show()
      return 0

if __name__ == "__main__":
    ps = np.linspace(0, 0.5, 6)
    for p in ps:
        main(p)
