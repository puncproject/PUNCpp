import numpy as np
import matplotlib.pyplot as plt

files = ['KE.txt', 'PE.txt','TE.txt']

data = []
for fname in files:
    tmp = []
    f = open(fname, 'r').readlines()
    for j in range(0,len(f)):
        tmp.append(f[j].split()[0])
    tmp = [float(x) for x in tmp]
    data.append(tmp)

KE = np.array(data[0])
PE = np.array(data[1])
TE = np.array(data[2])

plt.plot(KE,label="Kinetic Energy")
plt.plot(PE,label="Potential Energy")
plt.plot(KE+PE,label="Total Energy")
plt.legend(loc='best')
plt.grid()
plt.xlabel("Timestep")
plt.ylabel("Normalized Energy")

plt.show()

