from numpy import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
	for fname in fnames[:5]:
		with open(fname) as f:
			content = f.readlines()

		vs = [x.strip() for x in content]
		vs = array(vs).astype(float)
		vs = vs.reshape(-1, dim)
		print(vs.shape)
		plt.figure()
		plt.hist(vs[:,0], bins=300, normed=1)
		plt.figure()
		plt.hist(vs[:,1], bins=300, normed=1)
		if fname=="build/vs5.txt":
			plt.figure()
			plt.hist2d(vs[:,0],vs[:,1],bins=100,norm=LogNorm())
			plt.axis('equal')
		if dim==3:
			plt.figure()
			plt.hist(vs[:,2], bins=200)

	plt.show()
