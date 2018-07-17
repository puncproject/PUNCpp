#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import value as constants

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

Rp    = debye
Te    = (e*debye)**2*ne/(eps0*kB)
V0    = kB*Te/e;
I0    = -e*ne*Rp**2*np.sqrt(8*np.pi*kB*Te/me)

print('ne    =' , ne)
print('debye =' , debye)
print('wpe   =' , wpe)
print('vthe  =' , vthe)
print('vthi  =' , vthi)
print('Rp    =' , Rp)
print('Te    =' , Te)
print('V0    =' , V0)
print('I0    =' , I0)

data = []
with open('history.dat') as f:
    for l in f:
        data.append(l.split())

data = np.array(data, dtype=np.float)

plt.figure()
plt.plot(data[:,5])
plt.title('Potential')
plt.grid()
plt.show()

plt.figure()
plt.plot(data[:,6])
plt.plot(I0*np.ones(data[:,6].shape))
plt.title('Current')
plt.grid()
plt.show()
