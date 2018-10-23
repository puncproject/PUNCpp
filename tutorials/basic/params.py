#!/usr/bin/env python

from scipy.constants import value as constants
import numpy as np
from langmuir import *

# Constants
eps0 = constants('electric constant')
k    = constants('Boltzmann constant')
e    = constants('elementary charge')
m_e  = constants('electron mass')
m_p  = constants('proton mass')

# Probe input (spherical)
r = 1e-3 #0.255e-3
l = 50e-3

# Electron input
qe   = -e
me   = m_e
ne   = 10e10
eV   = 0.26
Te   = eV*e/k
vthe = np.sqrt(k*Te/me)
wpe  = np.sqrt(ne*qe**2/(eps0*me))
Le   = np.sqrt(eps0*k*Te/(qe**2*ne))

# Le = 1 # debye length
# vthe = wpe*Le
# Te = vthe**2*me/k

# Ion input
qi = e
mi = m_p
ni = ne
Ti = Te
vthi = vthe/np.sqrt(mi/me)

V   = 3
eta = e*V/(k*Te)
# eta = 25
# V   = eta*k*Te/e

R   = r/Le
I0  = lafr_norm_current('Cylinder', r, ne, Te)
f   = lafr_attr_current('Cylinder')
Ip  = I0*f(R, eta)
I   = Ip*l

plasma = []
plasma.append(Species('electron', n=ne, T=Te))
plasma.append(Species('proton',   n=ni, T=Ti))
I_OML = OML_current(Cylinder(r, 1), plasma, V)
I_OML_el = OML_current(Cylinder(r, 1), plasma[0], V)


# Probe, derived
# V0 = k*Te/np.abs(qe)
# I0 = qe*ne*r**2*np.sqrt(8*np.pi*k*Te/me)
# I = 2.945*I0
# V = 2*V0

keys = list(globals().keys())
keys = list(filter(lambda x: not x.startswith('_'), keys))
for key in keys:
    print('{:10} = {}'.format(key,globals()[key]))
