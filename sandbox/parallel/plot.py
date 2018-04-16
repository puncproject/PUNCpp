import numpy as np
import matplotlib.pyplot as plt

files = ['build/error.txt', 'build/h.txt']
err = []
h = []
N = [4,8,16,32,64]
f = open(files[0], 'r').readlines()
for j in range(0,len(f)):
    w = f[j].split()
    err.append(w[0])


f = open(files[1], 'r').readlines()
for j in range(0,len(f)):
    w = f[j].split()
    h.append(w[0])

err = np.array(err, dtype=float)
h = np.array(h, dtype=float)
print("size: ", len(err), len(h))
len1 = len(err)

for i in range(len1):
    order = np.log(err[i]/err[i-1])/np.log(h[i]/h[i-1]) if i>0 else 0
    print("Running with N=%3d: h=%2.2E, E=%2.2E, order=%2.2E"%(N[i],h[i],err[i],order))

"""
for i in range(len1):
    order = np.log(err2[i]/err2[i-1])/np.log(h[i]/h[i-1]) if i>0 else 0
    print("Running with N=%3d: h=%2.2E, E=%2.2E, order=%2.2E"%(N[i],h[i],err2[i],order))
"""

y = np.array([x**2 for x in h])
plt.loglog(1./h,err)
#plt.loglog(h,err2, '*k')
plt.loglog(1./h,y,'--r')
plt.grid()
plt.title('Convergence of PUNC PoissonSolver class')
plt.xlabel('Minimum cell diameter in mesh')
plt.ylabel('L2 error norm')
plt.show()
