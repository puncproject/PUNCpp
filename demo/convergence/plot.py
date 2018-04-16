import numpy as np
import matplotlib.pyplot as plt

files = ['build/phi.txt', 'build/E.txt', 'build/h.txt']
err_f, err_e = [], []
h = []

f = open(files[0], 'r').readlines()
for j in range(0,len(f)):
    w = f[j].split()
    err_f.append(w[0])

f = open(files[1], 'r').readlines()
for j in range(0, len(f)):
    w = f[j].split()
    err_e.append(w[0])

f = open(files[2], 'r').readlines()
for j in range(0,len(f)):
    w = f[j].split()
    h.append(w[0])

err_f = np.array(err_f, dtype=float)
err_e = np.array(err_e, dtype=float)
h = np.array(h, dtype=float)
print("size: ", len(err_f), len(h))
len1 = len(err_f)

for i in range(len1):
    order_f = np.log(err_f[i]/err_f[i-1])/np.log(h[i]/h[i-1]) if i>0 else 0
    order_e = np.log(err_e[i]/err_e[i-1])/np.log(h[i]/h[i-1]) if i>0 else 0
    print("Running with h=%2.2E, E_phi=%2.2E, order_phi=%2.2E"%(h[i],err_f[i],order_f))
    print("Running with h=%2.2E, E_E=%2.2E, order_E=%2.2E" %
          (h[i], err_e[i], order_e))

y = np.array([x**2 for x in h])
z = np.array([x for x in h])
plt.loglog(1./h,err_f, 'b')
plt.loglog(1./h,err_e, 'r')
plt.loglog(1./h,y,'--k')
plt.loglog(1. / h, z, '--k')
plt.grid()
plt.title('Convergence of PUNC PoissonSolver class')
plt.xlabel('Minimum cell diameter in mesh')
plt.ylabel('L2 error norm')
plt.show()
