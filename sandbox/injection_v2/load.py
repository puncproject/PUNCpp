from numpy import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import erfcinv, erfinv, erf, erfc

vth = 1.0
vd = array([1.0,0.0])
xs = linspace(vd[0] - 5 * vth, vd[0] + 5 * vth, 1000)

def pdf_maxwellian(i, t):
	return 1.0 / (sqrt(2 * pi) * vth) *\
		exp(-0.5 * ((t - vd[i])**2) / (vth**2))

def pdf_flux_nondrifting(s, t):
	return s * (t / vth**2) * exp(-0.5 * (t / vth)**2)

def pdf_flux_drifting(i, s, t):
	return (s * t * exp(-0.5 * ((t - vd[i])**2) / (vth**2))) /\
		(vth**2 * exp(-0.5 * (vd[i] / vth)**2) +
			s * sqrt(0.5 * pi) * vd[i] * vth *
			(1. + s * erf(vd[i] / (sqrt(2) * vth))))

case = 2
dim = 2
if case ==1:
	fname ="vs.txt"
	with open(fname) as f:
		content = f.readlines()

	vs = [x.strip() for x in content]
	vs = array(vs).astype(float)

	if dim==1:
		print(vs.shape)
		plt.hist(vs, bins=100)
	else:
		vs = vs.reshape(-1,dim)
		print(vs.shape)
		plt.hist(vs[:,0], bins=300, color='blue', normed=1)
		plt.figure()
		plt.hist(vs[:,1], bins=300, color='blue', normed=1)
		# plt.hist2d(vs[:,0],vs[:,1],bins=100,norm=LogNorm())
		# plt.axis('equal')
	plt.show()
elif case == 2:

	fnames = ["build/vs1.txt", "build/vs2.txt", "build/vs3.txt", "build/vs4.txt",
			  "build/vs5.txt", "build/vs6.txt"]
	for i, fname in enumerate(fnames[:2*dim]):
		with open(fname) as f:
			content = f.readlines()

		vs = [x.strip() for x in content]
		vs = array(vs).astype(float)
		vs = vs.reshape(-1, dim)
		print(vs.shape)
		plt.figure()
		plt.hist(vs[:, 0], bins=300, normed=1)
		if i==0:
			plt.plot(xs, pdf_flux_drifting(0,1, xs), color='red')
		if i==1:
			plt.plot(xs, pdf_maxwellian(0, xs), color='red')
		if i==2:
			plt.plot(xs, pdf_flux_drifting(0,-1, xs), color='red')
		if i==3:
			plt.plot(xs, pdf_maxwellian(0, xs), color='red')
		plt.figure()
		plt.hist(vs[:, 1], bins=300, normed=1)
		if i == 0:
			plt.plot(xs, pdf_maxwellian(1, xs), color='red')
		if i==1:
			plt.plot(xs, pdf_flux_nondrifting(1, xs), color='red')
		if i==2:
			plt.plot(xs, pdf_maxwellian(1, xs), color='red')
		if i==3:
			plt.plot(xs, pdf_flux_nondrifting(-1, xs), color='red')

            # plt.hist(vs[:,0], bins=300, color = 'blue', normed=1)
            # 
            # # plt.plot(xs, pdf_flux_nondrifting(1,xs), color='red')
            # plt.figure(figsize=(6, 5))
            # plt.hist(vs[:,1], bins=300, color = 'blue', normed=1)
            # plt.plot(xs, pdf_maxwellian(1,xs), color='red')

		if fname == "build/vs5.txt":
			plt.figure()
			plt.hist2d(vs[:, 0], vs[:, 1], bins=100, norm=LogNorm())
			plt.axis('equal')
		if dim == 3:
			plt.figure()
			plt.hist(vs[:, 2], bins=200)

	plt.show()
