#!/usr/bin/env python

from scipy.constants import value as constants
import numpy as np

# Constants
eps0 = constants('electric constant')
k    = constants('Boltzmann constant')
e    = constants('elementary charge')
m_e  = constants('electron mass')
m_p  = constants('proton mass')

# Probe input (spherical)
R = 1

# Electron input
qe = -e
me = m_e
ne = 1e10
Le = 1 # debye length

# Electron, derived
wpe = np.sqrt(ne*qe**2/(eps0*me))
vthe = wpe*Le
Te = vthe**2*me/k

# Ion input
qi = e
mi = m_p
ni = ne
Ti = Te

mi_e = mi/m_e

# Ion, derived
vthi = vthe/np.sqrt(mi/me)

# Probe, derived
V0 = k*Te/np.abs(qe)
I0 = qe*ne*R**2*np.sqrt(8*np.pi*k*Te/me)
I = 2.945*I0
V = 2*V0

keys = list(globals().keys())
keys = list(filter(lambda x: not x.startswith('_'), keys))
for key in keys:
    print('{:10} = {}'.format(key,globals()[key]))
