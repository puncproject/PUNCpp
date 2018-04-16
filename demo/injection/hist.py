from numpy import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

fnames = ["build/vels_pre.txt", "build/vels_post.txt"]

vth = 1.0
vd = array([0.0,0.0])
xs = linspace(vd[0] - 5 * vth, vd[0] + 5 * vth, 1000)

def pdf_maxwellian(i, t):
    return 1.0 / (sqrt(2 * pi) * vth) *\
        exp(-0.5 * ((t - vd[i])**2) / (vth**2))

dim = 2
for j, fname in enumerate(fnames):
	f = open(fname, "r")
	v = []
	for i,line in enumerate(f):
		tmp = [float(tmp1) for tmp1 in line.split('\n')[0].split()]
		v.append(tmp[0:dim])

	v = array(v)
	print(v.shape)
	plt.figure()
	plt.hist2d(v[:,0],v[:,1],bins=100,norm=LogNorm())
	plt.figure()
	plt.hist(v[:,0],bins=300, color = 'blue', normed=1)
	plt.plot(xs, pdf_maxwellian(0,xs), color='red')
	plt.figure()
	plt.hist(v[:,1],bins=300, color = 'blue', normed=1)
	plt.plot(xs, pdf_maxwellian(1,xs), color='red')
	if dim==3:
		plt.figure()
		plt.hist(v[:,2],bins=300, color = 'blue', normed=1)
plt.show()
