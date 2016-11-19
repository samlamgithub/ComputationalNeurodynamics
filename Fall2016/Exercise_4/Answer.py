# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Computational Neurodynamics
Exercise 4 (Course Work)

Jiahao Lin 2016
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rn
from IzNetwork import IzNetwork


def BackgroundNoise(ExiNumPerMod, moduleNum):
    lamda = 0.01
    samples = rn.binomial(1, lamda, ExiNumPerMod*moduleNum)
    currents = np.zeros([1000, ])
    currents[0:ExiNumPerMod*moduleNum] = map(lambda x: 15.0 if x > 0.0
                                             else 0.0, samples)
    return currents


def BuildExcitoryModule(ExiNumPerMod, ExiModConnNum, EtoEW):
    Module = np.zeros([ExiNumPerMod, ExiNumPerMod])
    connections = rn.choice(10000, size=ExiModConnNum, replace=False)
    for i in range(ExiModConnNum):
        connection = connections[i]
        xindex = connection / ExiNumPerMod
        yindex = connection % ExiNumPerMod
        if xindex != yindex:
            Module[xindex, yindex] = EtoEW()
    return Module


def BuildNetwork(rewiringProb):
    """
    Create a network with ExiNumrewiringProberMod*moduleNum excitatory
    Izhikevich neurons with 200 Inhibitory neurons, respectively.
    """
    moduleNum = 8
    ExiNumPerMod = 100
    ExiModConnNum = 1000
    InhiNum = 200
    InhiToExciNum = 4
    N = moduleNum * ExiNumPerMod + InhiNum
    Dmax = 20  # Max synaptic delay, in ms
    net = IzNetwork(N, Dmax)

    EtoESF = 17.0
    EtoISF = 50.0
    ItoESF = 2.0
    ItoISF = 1.0

    EtoEW = lambda: 1.0

    EtoIW = lambda: rn.rand()

    ItoEW = lambda: -rn.rand()

    ItoIW = lambda: -rn.rand()

    # Build Excitatory modules
    Modules = np.zeros([moduleNum, ExiNumPerMod, ExiNumPerMod])
    EtoEMatrix = np.zeros([ExiNumPerMod*moduleNum, ExiNumPerMod*moduleNum])
    for i in range(moduleNum):
        Module = BuildExcitoryModule(ExiNumPerMod, ExiModConnNum, EtoEW)
        Modules[i, :, :] = Module
        startX = i * ExiNumPerMod
        endX = startX + ExiNumPerMod
        EtoEMatrix[startX: endX, startX: endX] = Module

    # Rewiring Excitatory modules
    EtoEMx = EtoEMatrix[0: ExiNumPerMod*moduleNum, 0: ExiNumPerMod*moduleNum]
    for i in range(moduleNum):
        originalModule = Modules[i]
        for xCor in range(ExiNumPerMod):
            for yCor in range(ExiNumPerMod):
                if originalModule[xCor, yCor] > 0.0:
                    if rn.rand() < rewiringProb:
                        moduleIndex = rn.randint(0, 8, dtype="int")
                        while moduleIndex == i:
                            moduleIndex = rn.randint(0, 8, dtype="int")
                        NeuronIndex = rn.randint(0, 100, dtype="int")
                        xIndex = i*ExiNumPerMod + xCor
                        yIndex = i*ExiNumPerMod + yCor
                        # Delete original Wire
                        EtoEMx[xIndex, yIndex] = 0.0
                        # Rewire
                        newYIndex = moduleIndex * ExiNumPerMod + NeuronIndex
                        EtoEMx[xIndex, newYIndex] = EtoEW()

    EtoEMatrix[0: ExiNumPerMod*moduleNum, 0: ExiNumPerMod*moduleNum] = EtoEMx

    # Connect Excitatory to Inhibitory Neurons
    EtoIMatrix = np.zeros([ExiNumPerMod*moduleNum, InhiNum])
    randomInhiIndex = rn.choice(InhiNum, size=InhiNum, replace=False)
    for inhibitory in range(InhiNum):
        index = randomInhiIndex[inhibitory]
        for j in range(InhiToExciNum):
            EtoIMatrix[4 * index + j, inhibitory] = EtoIW()

    ItoEMatrix = np.zeros([InhiNum, ExiNumPerMod*moduleNum])
    for i in range(InhiNum):
        for j in range(ExiNumPerMod*moduleNum):
            ItoEMatrix[i, j] = ItoEW()

    ItoIMatrix = np.zeros([InhiNum, InhiNum])
    for i in range(InhiNum):
        for j in range(InhiNum):
            ItoIMatrix[i, j] = ItoIW()

    # Build network as a block matrix. Block [i,j] is the connection from
    # neuron i to neuron j
    WeightMatrix = np.bmat([[EtoESF * EtoEMatrix, EtoISF * EtoIMatrix],
                            [ItoESF * ItoEMatrix, ItoISF * ItoIMatrix]])

    WeightMatrix = np.array(WeightMatrix)
    # Plot connectivity matrix
    plt.matshow(WeightMatrix, vmin=-2.0, vmax=50.0)
    plt.show()

    CDMatrix = np.ones([N, N], dtype="int")
    for i in range(N):
        for j in range(N):
            if WeightMatrix[i, j] > 0.0:
                if i < ExiNumPerMod*moduleNum and j < ExiNumPerMod*moduleNum:
                    randomDelay = rn.randint(0, 20, dtype="int")
                    CDMatrix[i, j] = CDMatrix[i, j] + randomDelay

    # Plot connection delay matrix
    # plt.matshow(CDMatrix, vmin=1.0, vmax=20.0)
    # plt.show()

    # All neurons are heterogeneous excitatory or inhibitory regular spiking
    As = np.zeros([1000, ])
    Bs = np.zeros([1000, ])
    Cs = np.zeros([1000, ])
    Ds = np.zeros([1000, ])
    for i in range(1000):
        #   print is
        r = rn.rand()
        if i < 800:
            # Exi
            Ea = 0.02
            Eb = 0.2
            Ec = -65 + 15*(r**2)
            Ed = 8 - 6*(r**2)
            As[i] = Ea
            Bs[i] = Eb
            Cs[i] = Ec
            Ds[i] = Ed
        else:
            # Inhibitory
            Ia = 0.02 + 0.08*r
            Ib = 0.25 - 0.05*r
            Ic = -65
            Id = 2
            As[i] = Ia
            Bs[i] = Ib
            Cs[i] = Ic
            Ds[i] = Id

    net.setWeights(WeightMatrix)
    net.setDelays(CDMatrix)
    net.setParameters(np.array(As), np.array(Bs), np.array(Cs), np.array(Ds))

    return net


def runSimulation(rewiringProb):
    """
    Create and run the network with constant input.
    """
    moduleNum = 8
    ExiNumPerMod = 100
    print "rewiringProb: ", rewiringProb

    Tmin = 0
    Tmax = 1000

    # Construct the network
    net = BuildNetwork(rewiringProb)
    # Set current and initialise arrays
    I = BackgroundNoise(ExiNumPerMod, moduleNum)
    T = np.arange(Tmin, Tmax + 1)
    Rx = np.array([])
    Ry = np.array([])
    MeanFireX = [[]] * 8
    MeanFireY = [[]] * 8

    # Simulate
    c1 = 0
    c2 = 0
    for t in xrange(len(T)):
        I = BackgroundNoise(ExiNumPerMod, moduleNum)
        net.setCurrent(I)
        fIdx = net.update()
        # print "index count: ", len(fIdx)
        for k in range(len(fIdx)):
            idx = fIdx[k]
            c1 += 1
            if idx < 800:
                c2 += 1
                Rx = np.append(Rx, t)
                Ry = np.append(Ry, idx)
                MeanFireX[idx/100] = np.append(MeanFireX[idx/100], t)
                MeanFireY[idx/100] = np.append(MeanFireY[idx/100], idx)

    print "len: ",  Rx.shape, Ry.shape, c1, c2

    MeanFireXXs = [[]] * 8
    MeanFireYYs = [[]] * 8
    for mInx in range(8):
        Xc = MeanFireX[mInx]
        for windowIndex in range(50):
            startT = 20 * windowIndex
            endT = startT + 50
            countPoint = 0
            for v in range(len(Xc)):
                if Xc[v] >= startT and Xc[v] < endT:
                    countPoint += 1
            MeanFireXXs[mInx] = np.append(MeanFireXXs[mInx], startT + 25)
            MeanFireYYs[mInx] = np.append(MeanFireYYs[mInx], countPoint/50.0)

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
    ax1.set_title('Time(ms) + 0s')

    plt.show()
    return 0

if __name__ == "__main__":
    rewiringProbabilities = np.linspace(0, 0.5, 6)
    for rewiringP in rewiringProbabilities:
        runSimulation(rewiringP)
