from numpy import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

case = 1

if case ==1:
	fname ="vs.txt"

	dim = 1
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
	fnames = ["build/vs1.txt", "build/vs2.txt", "build/vs3.txt", "build/vs4.txt"]
	for fname in fnames:
		with open(fname) as f:
			content = f.readlines()

		vs = [x.strip() for x in content] 
		vs = array(vs).astype(float)
		vs = vs.reshape(-1, 2)
		print(vs.shape)
		plt.figure()
		plt.hist(vs[:,0], bins=100)
		plt.figure()
		plt.hist(vs[:,1], bins=100)

	plt.show()
