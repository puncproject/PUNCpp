from numpy import *
import matplotlib.pyplot as plt

fnames = ["pos.txt", "cg_pos.txt"]
labels = ["Dolfin", "CG1"]
for i, fname in enumerate(fnames):
	pos = []
	f = open(fname, 'r').readlines()
	for j in range(0,len(f)):
		pos.append(f[j].split()[0])

	pos = array([float(x) for x in pos])
	pos = pos.reshape((-1,3))
	# fig = plt.figure()
	plt.plot(pos[:, 0], pos[:, 1], label=labels[i])
	plt.xlim([0, 2])
	plt.ylim([0, 2])
	plt.grid()
	plt.axis('equal')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.legend(loc='lower right')
	# plt.show()

plt.show()
