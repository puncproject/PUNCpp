#!/usr/bin/env python

# Temporary plotting file

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import value as constants
import sys

def expAvg(data, dt, tau=0.1):
    """
    Makes an exponential moving average of "data". dt is the timestep between
    each sample in some unit and tau is the relaxation time in the same unit.
    """
    weight = 1-np.exp(-dt/tau)
    result = np.zeros(data.shape)
    result[0] = data[0]
    for i in range(1,len(data)):
        result[i] = weight*data[i] + (1-weight)*result[i-1]
    return result

def plotAvg(x, y, label=None, tau=0.1):
    """
    Plots a moving exponential average of "y" versus "x" in a matplotlib plot
    while showing the raw values of "y" in the background. tau is the relaxation
    time in the same unit as the value on the x-axis.
    """
    dx = x[1]-x[0]

    if tau != 0.0:
        plt.plot(x, y, '#CCCCCC', linewidth=1, zorder=0)

    p = plt.plot(x, expAvg(y, dx, tau), linewidth=1, label=label)

    # returning this allows color to be extracted
    return p


eps0 = constants('electric constant')
e    = constants('elementary charge')
me   = constants('electron mass')
mp   = constants('proton mass')
kB   = constants('Boltzmann constant')

ne    = 1e10
debye = 1
wpe   = np.sqrt(e**2*ne/(eps0*me))
vthe  = wpe*debye
vthi  = vthe*np.sqrt(me/mp)

Rp    = 1*debye
Te    = me*vthe**2/kB
V0    = kB*Te/e;
I0    = -e*ne*Rp**2*np.sqrt(8*np.pi*kB*Te/me)
# Vl = 25*V0
# Il = 6.053*I0
Vl    = 2*V0
Il = np.finfo(float).eps * I0

data = []
if len(sys.argv)>1:
    fname = sys.argv[1]
else:
    fname = 'history.dat'

with open(fname) as f:
    for l in f:
        if l[0] != '#': data.append(l.split())

data = np.array(data, dtype=np.float)

tau = 1
xaxis = data[:,1]*1e6

V = data[:,6]
I = data[:,7]

dx = xaxis[1]-xaxis[0]
Vm = expAvg(V, dx, tau)[-1]
Im = expAvg(I, dx, tau)[-1]

current_error = np.abs((Im - Il) / Il) if (Im -
                                           Il) > np.finfo(float).eps else np.finfo(float).eps
print("Voltage error: {}".format(np.abs((Vm-Vl)/Vl)))
print("Current error: {}".format(current_error))

plt.figure()
plotAvg(xaxis, V, tau=tau)
plt.plot(xaxis,Vl*np.ones(xaxis.shape),'k:')
plt.title('Potential')
plt.xlabel('us')
plt.ylabel('V')
plt.grid()
# plt.show()

plt.figure()
plotAvg(xaxis, I, tau=tau)
plt.plot(xaxis,Il*np.ones(xaxis.shape),'k:')
plt.title('Current')
plt.xlabel('us')
plt.ylabel('A')
plt.grid()
plt.show()

