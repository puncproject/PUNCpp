from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import gamma
from scipy.constants import value as constants

fnames = ["vels_pre.txt", "vels_post.txt"]

eps0 = constants('electric constant')
e = constants('elementary charge')
me = constants('electron mass')
mp = constants('proton mass')
kB = constants('Boltzmann constant')

ne = 1e10
debye = 1
wpe = np.sqrt(e**2 * ne / (eps0 * me))
vth = wpe * debye
print(vth)
# vthi = vthe * np.sqrt(me / mp)

vd = array([0.0,0.0])
k = 3.0
alpha = 1.0
D = 1
pdf_type = "kappa"# "kappa", "cairns", "maxwellian"
dim = 3

xs = linspace(vd[0] - 5 * vth, vd[0] + 5 * vth, 1000)

def pdf_cairns(v):
	return 1.0/((1+15*alpha)*(2*np.pi*vth**2)**(D/2.0))*\
		(1. + (D**2-8*D+15)*alpha + 2*alpha*(3.-D)*v**2/vth**2 +\
		(alpha/vth**4)* v**4) *\
		np.exp(-0.5 *v**2 / (vth**2))

def pdf_kappa_old(v):
	return 1.0 / ((np.pi * (2 * k - 3.) * vth**2)**(D / 2.0)) *\
		((gamma(k + 0.5 * (D - 1.0))) / (gamma(k - 0.5))) *\
		(1. + v**2 / ((2 * k - 3.) * vth**2))**(-(k + 0.5 * (D - 1.)))

def pdf_kappa(v):
	return 1.0 / ((np.pi * (2 * k - D) * vth**2)**(D / 2.0)) *\
		((gamma(k + 1.0)) / (gamma(k + 0.5*(2.-D)))) *\
		(1. + v**2 / ((2 * k - D) * vth**2))**(-(k + 1.))

def pdf_kappa_cairns(v):
	return (1.0 / ((np.pi * (2 * k - 3.) * vth**2)**(D / 2.0)) )*\
		(1.0/(1 + 15 * alpha*((2.0*k-3.0)/(2.0*k-3.0))))*\
		((gamma(k + 0.5 * (D - 1.0))) / (gamma(k - 0.5))) *\
		(1.0+alpha*v**4/vth**4)*\
		(1. + v**2 / ((2 * k - 3.) * vth**2))**(-(k + 0.5 * (D - 1.)))

def pdf_maxwellian(i, t):
    return 1.0 / (sqrt(2 * pi) * vth) *\
        exp(-0.5 * ((t - vd[i])**2) / (vth**2))


for j, fname in enumerate(fnames):
	f = open(fname, "r")
	v = []
	for i,line in enumerate(f):
		tmp = [float(tmp1) for tmp1 in line.split('\n')[0].split()]
		v.append(tmp[0:dim])

	v = array(v)
	print(v.shape)
	plt.figure()
	plt.hist2d(v[:,0],v[:,1],bins=300,norm=LogNorm())
	
	plt.figure()
	plt.hist(v[:,0],bins=200, color = 'blue', normed=1)
	if pdf_type=="maxwellian":
		plt.plot(xs, pdf_maxwellian(0,xs), color='red')
	elif pdf_type == "cairns":
		plt.plot(xs, pdf_cairns(xs), color='red')
	elif pdf_type=="kappa":
		plt.plot(xs, pdf_kappa(xs), color='red')
	elif pdf_type == "kappa_cairns":
		plt.plot(xs, pdf_kappa_cairns(xs), color='red')
	plt.figure()
	plt.hist(v[:,1],bins=200, color = 'blue', normed=1)
	if pdf_type == "maxwellian":
		plt.plot(xs, pdf_maxwellian(1, xs), color='red')
	elif pdf_type == "cairns":
		plt.plot(xs, pdf_cairns(xs), color='red')
	elif pdf_type == "kappa":
		plt.plot(xs, pdf_kappa(xs), color='red')
	elif pdf_type == "kappa_cairns":
		plt.plot(xs, pdf_kappa_cairns(xs), color='red')

	if dim==3:
		plt.figure()
		plt.hist(v[:,2],bins=200, color = 'blue', normed=1)
		if pdf_type == "maxwellian":
			plt.plot(xs, pdf_maxwellian(2, xs), color='red')
		elif pdf_type == "cairns":
			plt.plot(xs, pdf_cairns(xs), color='red')
		elif pdf_type == "kappa":
			plt.plot(xs, pdf_kappa(xs), color='red')
		elif pdf_type == "kappa_cairns":
			plt.plot(xs, pdf_kappa_cairns(xs), color='red')
plt.show()
